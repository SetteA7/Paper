# Paper: remote braking over a lossy network

ROS 2 (Jazzy) workspace for the braking experiment: a ROSbot streams its LiDAR to a
remote **controller**, which decides whether to brake and sends `/cmd_vel` back over a
wireless link. The brake threshold can be fixed or adapted to the measured network
latency and loss (SMART / SUPERSMART). Runs can be fully automated and are logged to CSV.

| package | what it is |
|---|---|
| `link_monitor_msg` | message types for the link monitor heartbeats (`LinkState`) |
| `monitor_pkg` | the nodes: `controller_monitor`, `bot_monitor`, `noise` |

## Build

Clone this repo as the `src` folder of a workspace, then:

```bash
cd ~/ros2_ws_A7
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

With `--symlink-install`, edits to the Python files take effect without rebuilding.
Rebuild only after adding a new file or entry point (`setup.py`).

Both machines must use the same `ROS_DOMAIN_ID` and middleware, and their clocks
should be synced (chrony): one-way latencies are measured across the two clocks.

## Quick start

```bash
# on the bot (keep it running; it survives controller restarts)
ros2 run monitor_pkg bot_monitor

# on the controller: one automated run, fixed threshold
ros2 run monitor_pkg controller_monitor
```

## `bot_monitor` (run on the bot)

The bot side of the link monitor. Every 0.1 s, both ends send a heartbeat on their own topic
(`network_downlink` from the bot, `network_uplink` from the controller), and each end
measures the direction it receives: latency (median of the last 100 packets) and
loss (missing sequence numbers over the last 100). The bot reports its uplink
measurements back in its heartbeats. When the controller restarts, the bot detects
it and resets its statistics.

## `controller_monitor` (run on the controller)

```bash
ros2 run monitor_pkg controller_monitor [FIXED|SMART|SUPERSMART] [RUNS=n | MANUAL] [CSV|NOCSV]
```

| option | meaning |
|---|---|
| `FIXED` (default) | brake threshold S = `SAFE_THRESHOLD` (75 cm). `NOSMART` is an alias |
| `SMART` | S_high: budget `T*(N*+1)`, meets `P_TARGET` for any wait times (conservative) |
| `SUPERSMART` | S_smart: budget `B_min`, the lowest S whose exact crash probability is `P_TARGET` |
| `RUNS=n` | automated test, n runs (default 1) |
| `MANUAL` | old open-ended mode: drive and brake until Ctrl+C, no return, no alignment |
| `CSV` / `NOCSV` | per-scan CSV on/off in `MANUAL` mode (default on); automated runs always save |

Examples:

```bash
ros2 run monitor_pkg controller_monitor SUPERSMART RUNS=20
ros2 run monitor_pkg controller_monitor SMART MANUAL
```

### Threshold model

`S(B) = VMAX/1000 * (B + tau_e2e) + L + m + d_brake`, where B is the time budget
(ms). `N*` is the first N >= 1 with F(N) <= `P_TARGET`. F(k) = P(at least k failed
transmissions) is computed from the uplink and downlink loss rates. The exact crash
probability `P_crash(B)` and `B_min` are implemented in
[`crash_model.py`](monitor_pkg/monitor_pkg/crash_model.py). In every mode, the
controller logs the exact `P_crash` of the threshold it is using.

### Automated runs (`RUNS=n`)

Each run repeats the manual procedure:

1. **align**: turns in place until perpendicular to the surface in front (line fit
   on the LiDAR points of that surface, tolerance `ALIGN_TOL_DEG` = 0.5 deg).
2. **wait**: stands still, waits for LiDAR and network data, measures the start distance.
3. **approach**: drives forward at `LINEAR_SPEED` with the normal braking logic,
   until the first brake caused by distance.
4. **stopping**: sends 0 for `STOP_SETTLE_S`, then measures where the bot stopped.
5. **return**: reverses (straight back) to the start distance, then the next run begins.

The session ends after n runs, or at the **first crash** (closest reading during a
run <= `CRASH_DIST_CM`). It also ends safely if alignment fails, the start position
is not reached in time, or something is closer than `REAR_STOP_CM` behind while
reversing. The log says which of these happened.

### Output

CSVs go to `~/ros2_ws_A7/csv/` (`CSV_DIR`):

- automated: `auto_<MODE>_<date_time>/run_001.csv`, `run_002.csv`, ... and `summary.csv`
- `MANUAL`: `robot_log_<date_time>.csv`

**Per-scan CSV** (one row per LiDAR scan, ~10 Hz; automated runs log approach and stop only):

| column | unit | meaning |
|---|---|---|
| `timestamp_lidar` | s | scan time stamped by the LiDAR driver (bot clock) |
| `timestamp_arrival` | s | when the scan reached the controller (controller clock) |
| `scan_delay_ms` | ms | arrival - lidar |
| `min_distance_cm` | cm | closest valid reading in the front sector (-1 = none) |
| `safe_threshold_cm` | cm | brake threshold S used for this scan |
| `brake_state` | 0/1 | 1 = brake |
| `tau_tx_ms`, `tau_rx_ms`, `tau_e2e_ms` | ms | uplink, downlink, round-trip latency |
| `p_tx`, `p_rx`, `p_e2e` | 0-1 | uplink, downlink, combined loss |
| `budget_ms` | ms | time budget B of the current threshold |
| `p_crash` | 0-1 | exact crash probability of the current threshold |

**summary.csv** (one row per run): `run, csv_file, start_dist_cm, brake_dist_cm,
safe_threshold_cm, N_at_brake, tau_e2e_ms_at_brake, p_e2e_at_brake, final_dist_cm,
min_dist_cm, stop_distance_cm, margin_to_crash_cm, crashed, mode,
budget_ms_at_brake, p_crash_at_brake, align_err_deg`.

### Settings

All settings are constants at the top of
[`controller.py`](monitor_pkg/monitor_pkg/controller.py):

- toggles: `DEFAULT_MODE`, `DEFAULT_RUNS`, `ENABLE_CSV`, `CSV_DIR`, `usePingTauRX`
  (`True`: tau_rx from the heartbeat ping; `False`: from the LiDAR scan delay)
- robot (measured in the lab): `M`, `BOT_LENGTH`, `D_BRAKE`, `VMAX`, `T`, `SAFE_THRESHOLD`,
  `LINEAR_SPEED` (keep it <= `VMAX`, or the model underestimates the risk)
- model: `P_TARGET`, `N_MAX`
- automated runs: `CRASH_DIST_CM`, `STOP_SETTLE_S`, `START_SETTLE_S`, `RETURN_*`, `REAR_STOP_CM`
- alignment: `ALIGN_*`. If the bot turns away from perpendicular, set `ALIGN_SIGN = -1`

## `noise` (network load generator)

Loads the link going **out** of the machine it runs on, with random-byte
messages (best effort, not retransmitted).

```bash
ros2 run monitor_pkg noise kbps=1024              # send 1.024 Mbit/s (and receive)
ros2 run monitor_pkg noise role=sink              # receive only
```

| option | default | meaning |
|---|---|---|
| `kbps=<n>` | 1024 | rate in kbit/s (1 kbit = 1000 bit) |
| `hz=<n>` | 100 | messages per second; size = kbps*1000/8/hz bytes (1024 kbps -> 1280 B) |
| `role=send\|sink` | send | `send` = send **and** receive, `sink` = receive only |
| `topic=<name>` | `network_noise` | must match on both machines |

DDS only sends data over the network when the other machine subscribes, so:

- **noise in one direction**: `noise kbps=...` on the sender, `noise role=sink` on the other machine
- **noise in both directions**: `noise kbps=...` on both machines (each is also the other's sink)

Every 2 s each node logs what it sent, how many remote receivers it has, and what it
received from the other machine, with loss. A node ignores its own messages.
`receivers 0` means nothing is going on the network.
