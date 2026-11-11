"""Background traffic (iperf3, UDP at a fixed rate) that loads the bot <-> controller link.

    ros2 run monitor_pkg controller_monitor ... KBPS=10 BOT=<bot address>
        10 kbit/s controller -> bot
    ros2 run monitor_pkg bot_monitor KBPS=1024
        1024 kbit/s bot -> controller

Who does what:
  - bot_monitor always runs two iperf3 servers (PORT_TX, PORT_RX) and publishes its KBPS
    on RATE_TOPIC (latched).
  - the controller runs both iperf3 clients: PORT_TX in normal mode (it sends,
    controller -> bot) and PORT_RX in reverse mode -R (the bot's server sends at the rate
    the bot asked for, bot -> controller). So starting and stopping the load is a local
    process start/stop on the controller: no stop message has to cross the loaded link.
  - AutoTest turns the load on in the wait phase before each measured run (not the
    calibration run 0), waits LOAD_WARMUP_S so the heartbeat windows see the loaded link,
    and turns it off when the run ends (before the bot drives back to its start).

Fail-safes: every client stops by itself after MAX_S, iperf3 gets SIGTERM if the
controller dies (PR_SET_PDEATHSIG), and the bot's server stops sending when the
controller's client goes away.

The bot must be reachable from the controller on TCP + UDP PORT_TX and PORT_RX, like the
zenoh bridge port. The controller needs no open port (UDP from -R comes back through NAT).
Each client writes iperf3's JSON log (per-second rate, jitter, lost packets in
`intervals`; a stop by SIGINT leaves "error": "interrupt ..." and an empty `end`).
"""
import ctypes
import shutil
import signal
import subprocess

from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

PORT_TX = 5211          # controller -> bot (not 5201: iperf3.service may hold it)
PORT_RX = 5212          # bot -> controller (iperf3 -R)
PACKET_BYTES = 1200     # UDP payload: below the path MTU, so iperf3 packets are not fragmented
MAX_S = 120             # a client stops by itself after this (a run takes far less)
CONNECT_TIMEOUT_MS = 3000

RATE_TOPIC = '/traffic_bot_kbps'    # std_msgs/Int32: bot -> controller rate asked by bot_monitor
# Latched: the controller gets the bot's rate whenever it starts
RATE_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def parse_kbps(flag: str, arg: str) -> int | None:
    """`KBPS=<n>` (any case) -> n, None for other flags."""
    if not flag.startswith('KBPS='):
        return None
    try:
        kbps = int(flag[5:])
    except ValueError:
        raise SystemExit(f"Bad option '{arg}': use KBPS=<kbit/s>")
    if kbps < 0:
        raise SystemExit('KBPS must be >= 0')
    return kbps


def _die_with_parent():
    """In the child: SIGTERM when the parent (controller / bot_monitor) dies."""
    ctypes.CDLL('libc.so.6', use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _iperf3(args: list) -> subprocess.Popen:
    if shutil.which('iperf3') is None:
        raise SystemExit('iperf3 not found: sudo apt install iperf3')
    return subprocess.Popen(['iperf3', *args], stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            preexec_fn=_die_with_parent)


def _stop(p: subprocess.Popen | None):
    if p is None or p.poll() is not None:
        return
    p.send_signal(signal.SIGINT)        # iperf3 writes its JSON log and exits
    try:
        p.wait(2.0)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


class LoadServers:
    """Bot side: the two iperf3 servers the controller's clients connect to."""

    def __init__(self):
        self.ports = [PORT_TX, PORT_RX]
        self.procs = [_iperf3(['-s', '-p', str(port)]) for port in self.ports]

    def check(self) -> list[str]:
        """Servers that exited since the last check (e.g. port already in use); dropped."""
        out = []
        for port, p in zip(self.ports, self.procs):
            if p.poll() is not None:
                err = p.stderr.read().decode(errors='replace').strip() if p.stderr else ''
                out.append(f'iperf3 server on port {port} exited ({p.returncode}): {err}')
        alive = [(port, p) for port, p in zip(self.ports, self.procs) if p.poll() is None]
        self.ports, self.procs = [a[0] for a in alive], [a[1] for a in alive]
        return out

    def stop(self):
        for p in self.procs:
            _stop(p)


class LoadClients:
    """Controller side: both directions' iperf3 clients, started and stopped per run."""

    def __init__(self, host: str, tx_kbps: int):
        self.host = host
        self.tx_kbps = tx_kbps      # controller -> bot (this side's KBPS=)
        self.rx_kbps = 0            # bot -> controller (bot_monitor's KBPS=, via RATE_TOPIC)
        self.procs = {}             # 'tx' / 'rx' -> Popen
        self.logs = {}              # 'tx' / 'rx' -> JSON log path
        self.on = (0, 0)            # (tx, rx) kbit/s of the load started last, (0, 0) if off

    @property
    def wanted(self) -> bool:
        return self.tx_kbps > 0 or self.rx_kbps > 0

    @property
    def active(self) -> bool:
        return bool(self.procs)

    def start(self, log_prefix: str) -> tuple[int, int]:
        """Start the configured directions; logs at <log_prefix>_tx.json / _rx.json.
        Returns the (tx, rx) kbit/s started."""
        self.stop()
        for d, kbps, port, extra in (('tx', self.tx_kbps, PORT_TX, []),
                                     ('rx', self.rx_kbps, PORT_RX, ['-R'])):
            if kbps <= 0:
                continue
            self.logs[d] = f'{log_prefix}_{d}.json'
            self.procs[d] = _iperf3([
                '-c', self.host, '-p', str(port), '-u', '-b', str(kbps * 1000),
                '-l', str(PACKET_BYTES), '-t', str(MAX_S),
                '--connect-timeout', str(CONNECT_TIMEOUT_MS),
                '-J', '--logfile', self.logs[d], *extra])
        self.on = (self.tx_kbps if 'tx' in self.procs else 0,
                   self.rx_kbps if 'rx' in self.procs else 0)
        return self.on

    def check(self) -> list[str]:
        """Clients that exited by themselves since the last check (server unreachable,
        MAX_S reached); they are dropped, so `on` shows what is still running."""
        out = []
        for d, p in list(self.procs.items()):
            if p.poll() is not None:
                out.append(f'iperf3 {d} client exited ({p.returncode}), see {self.logs[d]}')
                del self.procs[d]
        if out:
            self.on = (self.on[0] if 'tx' in self.procs else 0,
                       self.on[1] if 'rx' in self.procs else 0)
        return out

    def stop(self):
        for p in self.procs.values():
            _stop(p)
        self.procs = {}
        self.on = (0, 0)
