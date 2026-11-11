# Remote Braking over Starlink: Crash-Probability Experiments

Experiment code for the paper.

A ground robot drives toward a wall while it streams LiDAR scans over a **Starlink
satellite link** to a remote controller. The controller decides GO or STOP and sends the
command back over the same link. Packet loss and latency on the link delay the STOP, so
the robot may end up too close to the wall.

This repository contains the code used to measure that effect and to test the paper's
model of the crash probability:

- the **controller** computes a braking threshold S from live network statistics and
  logs every quantity of the model;
- the **bot monitor** measures the network from the robot's side and the real braking
  distance of every stop;
- an **automated test** (`AutoTest`) repeats the braking run many times, with a
  calibration run first, and writes one CSV per run plus a summary.

Target platform: Husarion ROSbot with an RPLIDAR on `sllidar_ros2`, ROS 2 Jazzy,
Python 3.12.

---

## Repository layout

| Path | What it is |
|---|---|
| `monitor_pkg/monitor_pkg/controller.py` | Controller node (`controller_monitor`): threshold, GO/STOP, ψ, CSV logging, `AutoTest` |
| `monitor_pkg/monitor_pkg/crash_model.py` | The paper's crash model (crash probability, S.M.A.R.T. threshold), pure math, no ROS |
| `monitor_pkg/monitor_pkg/client.py` | Bot node (`bot_monitor`): network server + `BrakeMonitor` (d_brake per stop) |
| `monitor_pkg/monitor_pkg/network_link.py` | Heartbeat link monitor used on both ends (loss, one-way latency) |
| `link_monitor_msg/` | Message package: `LinkState` (heartbeat). `LinkMetrics` is defined but unused |

Branches: `main` is the current design. `T-on-bot` holds an older design with a relay
timer on the bot, and `noise` holds an uplink noise generator.

---

## How it works

### Nodes and topics

The bot and the controller are two machines connected through the Starlink link. All
five topics below cross it.

| Topic | Type | From → to | QoS | Content |
|---|---|---|---|---|
| `/scan` | `sensor_msgs/LaserScan` | LiDAR driver (bot) → controller, BrakeMonitor | best effort (subscribers) | one scan per rotation, ~10 Hz |
| `/cmd_vel` | `geometry_msgs/TwistStamped` | controller → motor driver, BrakeMonitor | reliable publisher, depth 10 | GO = 0.5 m/s, STOP = 0, every T = 100 ms |
| `network_downlink` | `link_monitor_msg/LinkState` | bot → controller | best effort, depth 1 | heartbeat every 0.1 s, carries the uplink stats measured on the bot |
| `network_uplink` | `link_monitor_msg/LinkState` | controller → bot | best effort, depth 1 | heartbeat every 0.1 s |
| `/brake_distance` | `std_msgs/Float32` | BrakeMonitor → controller | reliable, depth 10 | measured d_brake (cm), once per STOP |

> **QoS matters for the model.** The model assumes that a lost scan or a lost STOP is
> replaced by the next one, one period T later (best-effort behaviour). With reliable QoS,
> DDS retransmits on its own instead, and loss turns into latency. A best-effort publisher
> cannot connect to a reliable subscriber. Keep this in mind when you change any topic.

### Naming: paper vs code

| | Paper | Code (`tx` / `rx`) |
|---|---|---|
| bot → controller (scans) | TX: `p_tx`, `τ_tx` | downlink: `p_rx`, `tau_rx` |
| controller → bot (STOP) | RX: `p_rx`, `τ_rx` | uplink: `p_tx`, `tau_tx` |

The crash probability is symmetric in the two losses, so the swap does not change it.

### The control loop

The controller runs everything on one `SingleThreadedExecutor`:

- **`scan_callback`** (every scan) records the arrival time, takes `min_dist` = closest
  valid ray of the 20 front rays, recomputes S, B and P_crash, and sets the brake flag
  `min_dist < S`.
- **`publish_cmd`** (timer, every T = 100 ms) measures ψ, then sends STOP or GO. It sends
  STOP if the flag is set, or if there is no scan, a stale scan (more than 0.5 s old) or
  an invalid scan.
- **`on_link_update`** (every 0.1 s) receives loss and latency of both directions from
  the network node and updates N\*.

---

## The model

A crash (margin breach) happens when the threshold transmission takes longer than the
time budget:

$$t_{e2e,thresh} = \varphi + \tau_{tx} + \psi + \tau_{rx} + (X+Y)\,T > \Delta t$$

| Symbol | Meaning | Where it comes from |
|---|---|---|
| φ | wait from the threshold crossing to the next LiDAR sample, U(0, T) | random start position (AutoTest) |
| τ_tx, τ_rx | one-way latencies (paper naming) | network node (ping), or the scans (`USEPING=False`) |
| ψ | wait at the controller, from scan arrival to the tick that sends the command | measured on every tick (`_measure_psi`) |
| X, Y | failed scan / STOP transmissions, geometric | loss from the heartbeats; scan loss X from the whole scans (`p_scan`) with `USEPING=False` |
| T | LiDAR period = control period, 100 ms | constant |

With $\Delta t = 1000\,(S - L - M - d_{brake})/v_{max}$ and the budget $B = \Delta t - \tau_{e2e} - \psi$:

$$P_{crash}(B) = F(n) - f(n)\,\frac{R}{T},\qquad n = \lfloor B/T \rfloor,\; R = B - nT$$

Here $F(k) = P(X+Y \ge k)$ and $f(k) = P(X+Y = k)$ (see `crash_model.py`).

### Threshold modes

| Mode | Threshold |
|---|---|
| `FIXED` (default) | S = `SAFE_THRESHOLD` = 75 cm |
| `SMART` | S = v·(T·(N\*+1) + τ_e2e) + L + M + d_brake. This is S_high with ψ replaced by T. N\* = first N with F(N) ≤ `P_TARGET` |
| `SUPERSMART` | S = v·(B_min + τ_e2e + ψ) + L + M + d_brake, the lowest S with P_crash ≤ `P_TARGET`. B_min = nT + T·(F(n) − P_TARGET)/f(n), n = N\* − 1 |

In every mode, B and the exact P_crash of the current S are computed and logged.

---

## Setup

Both machines (bot and controller) need this repository as `src/` of a colcon workspace.

```bash
mkdir -p ~/ros2_ws_A7 && cd ~/ros2_ws_A7
git clone git@github.com:SetteA7/Paper.git src
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install        # build from the workspace root, not from src/
source install/setup.bash
```

Requirements: ROS 2 Jazzy, Python 3.12 with `numpy`, and the robot's `sllidar_ros2`
driver on the bot.

- **Symlink install.** With `--symlink-install`, edits to the Python files take effect
  without rebuilding. Rebuild only when `setup.py` or the messages change.
- **Clocks.** One-way latencies, `scan_delay_ms` and `tau_rx_scan_ms` compare the two
  machines' clocks, so sync them with **chrony** (`chronyc tracking` should show a small
  offset). τ_e2e is offset-free; ψ, d_brake and VMAX each use one clock and need no sync.
  NTP-style sync assumes equal delays both ways. If the satellite link's uplink and
  downlink delays differ, the clock offset is wrong by half the difference. That shifts
  latency from one direction to the other, but τ_e2e is unaffected.

---

## Running

**On the bot** (with the robot's drivers already running):

```bash
ros2 run monitor_pkg bot_monitor
```

**On the controller:**

```bash
ros2 run monitor_pkg controller_monitor [FLAGS]
```

| Flag | Effect | Default |
|---|---|---|
| `FIXED` \| `SMART` \| `SUPERSMART` | threshold mode (`NOSMART` = `FIXED`) | `FIXED` |
| `RUNS=n` | automated test: calibration run 0 + n runs | `RUNS=1` |
| `MANUAL` | open-ended mode: drive and brake until Ctrl+C, no return, no runs | off |
| `USEPING=True` \| `USEPING=False` | downlink latency and loss used by the model: ping (network node) or scans | `True` |
| `CSV` \| `NOCSV` | write the CSV files | `CSV` |

Examples:

```bash
ros2 run monitor_pkg controller_monitor SUPERSMART RUNS=20
ros2 run monitor_pkg controller_monitor SMART RUNS=10 USEPING=False
ros2 run monitor_pkg controller_monitor MANUAL NOCSV
```

The network monitor nodes accept ROS parameters, which must be the same on both ends:
`period` (0.1 s), `window` (100), `timeout` (0.5 s), `best_effort` (true). For example:
`--ros-args -p window:=200`.

The startup line shows the active configuration, for example
`START | mode SUPERSMART | model downlink tau/p from PING | ... | AUTO calibration run 0 (FIXED) + 20 runs`.

---

## Automated test (`RUNS=n`)

Each run goes through five phases:

1. **align**: turn in place until the bot faces the wall (|yaw| ≤ 0.5° held 0.6 s at rest).
2. **wait**: stand still 2 s, with fresh scans and a live link.
3. **approach**: drive with the normal GO/STOP logic until the first STOP caused by
   `min_dist < S`.
4. **stopping**: send 0 for 2 s, then wait (up to 4 s) for the bot's d_brake.
5. **return**: reverse to the start position of the next run.

**Calibration run 0** (`CALIBRATION_RUN = True`):
- Run 0 always brakes in **FIXED** mode (S = `SAFE_THRESHOLD`), whatever the session mode,
  so the calibration does not depend on the model. Runs 1..n use the session mode.
- Run 0 uses the `VMAX` and `D_BRAKE` constants.
- Its measured cruise speed and d_brake become VMAX and D_BRAKE for runs 1..n, which then
  stay fixed.
- Both values must be measured and fall within 20% (VMAX) and 50% (D_BRAKE) of the
  constants, otherwise the session ends.
- Run 0 is logged as `run_000.csv` and `run = 0`, `mode = FIXED` in `summary.csv`. **Leave it out of the
  analysis.**
- With `CALIBRATION_RUN = False`, VMAX and D_BRAKE are instead updated after every run
  (`VMAX_FROM_RUNS`, `DBRAKE_FROM_RUNS`).

**Random start.** Each run starts at start + 3 cm + U(0, VMAX·T). That range is one
period of travel, so the threshold is crossed at a uniform phase: φ ~ U(0, T). A span of
about 4.5 cm is expected and enough.

**The session ends** after n runs, or earlier on:
- a crash (closest reading ≤ L);
- a failed calibration;
- an alignment failure (30 s timeout, or the wall angle exceeds 15°);
- a return timeout (30 s);
- an obstacle less than 30 cm behind the bot.

**Discard by hand** any run where the stale-scan brake fired or the approach paused.
The stale-scan brake is an equipment safety measure and is not part of the model.

---

## Output

Files go to `~/ros2_ws_A7/csv/`:
- AutoTest: `auto_<MODE>_<YYYYmmdd_HHMMSS>/run_000.csv …` plus `summary.csv`;
- `MANUAL`: `robot_log_<time>.csv`.

### Per-scan CSV (one row per scan)

| Column | Meaning |
|---|---|
| `timestamp_lidar` | `header.stamp` of the scan (bot clock, s): time of the first ray |
| `timestamp_arrival` | controller clock when the scan reached `scan_callback` (s) |
| `scan_delay_ms` | arrival − stamp. Includes the whole rotation (~100 ms) |
| `scan_time_ms` | `msg.scan_time`: duration of the rotation |
| `min_distance_cm` | closest valid front ray (−1 if none) |
| `safe_threshold_cm` | S at this scan |
| `brake_state` | 1 = brake flag set |
| `tau_tx_ms`, `tau_rx_ms`, `tau_e2e_ms` | **network node (ping)** latencies: code uplink, downlink, sum |
| `p_tx`, `p_rx`, `p_e2e` | heartbeat loss: code uplink, downlink, combined |
| `budget_ms` | B = Δt − τ_e2e(model) − ψ |
| `p_crash` | exact crash probability at S |
| `vmax_cm_s` | VMAX in use |
| `psi_ms` | ψ used by the model: 90th percentile of the last 50 waits |
| `tau_rx_scan_ms` | downlink latency from the scans: median of (arrival − stamp − scan_time), last 100 |
| `tau_e2e_model_ms` | τ_e2e used by the model: ping, or ping uplink + `tau_rx_scan_ms` with `USEPING=False` |
| `psi_sample_ms` | measured wait of the command based on this scan (empty if a newer scan came first) |
| `p_scan` | downlink loss of whole scans: missing DDS publication sequence numbers of `/scan`, last 100 scans (`nan` if the RMW gives none). The model's p_rx with `USEPING=False` |

### `summary.csv` (one row per run)

| Column(s) | Meaning |
|---|---|
| `run`, `csv_file` | run number (0 = calibration) and its CSV |
| `start_dist_cm`, `start_target_cm` | measured start distance and the random target it was driven to |
| `brake_dist_cm`, `safe_threshold_cm` | reading that triggered the STOP, and S |
| `final_dist_cm`, `stop_distance_cm` | where the bot stopped; brake_dist − final |
| `min_dist_cm`, `margin_to_crash_cm`, `crashed` | closest reading; min − L; 1 if ≤ 0 |
| `N_at_brake`, `budget_ms_at_brake`, `p_crash_at_brake` | model values at the brake |
| `tau_e2e_ms_at_brake`, `p_e2e_at_brake` | network node (ping) values at the brake |
| `tau_source`, `tau_e2e_model_ms_at_brake` | `PING` or `SCAN` (source of the model's downlink τ and p), and the τ_e2e the model used |
| `psi_ms_at_brake`, `psi_stop_ms` | ψ used by the model; measured wait of the STOP itself |
| `VMAX_cm_s`, `D_BRAKE_cm`, `T_ms`, `L_cm`, `M_cm`, `V_CMD_cm_s` | parameters used in this run |
| `v_cruise_cm_s`, `cruise_scans` | cruise speed measured in this run, and how many scans it is based on |
| `d_brake_meas_cm` | braking distance measured by the bot in this run |
| `mode`, `align_err_deg` | threshold mode; residual wall angle after aligning |

A useful consistency check: `d_brake_meas_cm` ≈ `stop_distance_cm` − v·(`tau_e2e_ms_at_brake` + `psi_stop_ms`)/1000.

---

## How the measured values are computed

- **ψ.** `scan_callback` stores the arrival time on the monotonic clock. At the start of
  the next tick, `publish_cmd` takes now − arrival. Only the newest scan per tick gets a
  sample, and a tick without a new scan adds none. The model uses the 90th percentile of
  the last 50 samples, or T until 10 samples exist. When arrivals land next to a tick,
  jitter makes ψ jump between ≈ 0 and ≈ T; the 90th percentile then takes the late case.
- **Loss and latency (network node).** Each side publishes a heartbeat with a sequence
  number and its send time. Missing sequence numbers count as lost, over a window of 100.
  If nothing arrives for 0.5 s, one loss is counted per period. Latency is the median of
  receive − send time. The bot measures the uplink and sends the result back in its
  heartbeat.
- **τ from the scans.** `header.stamp` is the first ray, and the message leaves after the
  last ray. During an approach the closest front ray is the newest one, so its age on
  arrival is arrival − stamp − scan_time. On a clean link this matches the ping (≈ 0.5 ms).
- **min_dist.** Closest ray with range_min < r < range_max among the 10 rays at each end
  of the array (index 0 points straight ahead; `rear_min` uses the 20 middle rays).
- **VMAX.** `cruise_speed()` keeps one point (stamp, min_dist) per approach scan. It drops
  the acceleration (speed over two intervals below 0.9× the speed just before the brake)
  and takes the least-squares slope of the rest. It needs at least 4 cruise scans, and
  the run's value is rejected if the bot was still accelerating.
- **d_brake (bot).** d_brake = d(STOP) − d(standstill), all on the bot's clock:
  - d(STOP): a line fitted to the front readings of the last 0.35 s, evaluated at the
    moment the first STOP reached the bot;
  - d(standstill): the median of 5 readings within 0.5 cm.
  - **Ray timing:** `sllidar_ros2` with `angle_compensate` stores the rays newest first.
    So ray k was measured at `stamp + (n−1−k)·time_increment`, not `stamp + k·time_increment`
    (`RAYS_REVERSED = True` in `client.py`). Before this was handled, d_brake came out
    about v·scan_time ≈ 4.3 cm too small.
  - Each `BRAKE` log line shows the ray indices used and the d_brake the other timing rule
    would give, as a check.
- **Wall angle.** A line is fitted to the wall segment straight ahead: it grows from the
  centre ray until neighbouring rays jump by more than 5 cm, or up to ±30°, with one pass
  of outlier rejection. Then `wall_yaw = −atan(b)`.

---

## Main constants (`controller.py`, class `LaserProcessor`)

| Constant | Value | Meaning |
|---|---|---|
| `T` | 100 ms | control period (= LiDAR period) |
| `LINEAR_SPEED` | 0.5 m/s | commanded speed (the bot reaches ~43–45 cm/s) |
| `VMAX`, `D_BRAKE` | 43.5 cm/s, 12.33 cm | starting values; AutoTest calibrates them in run 0 |
| `BOT_LENGTH` (L), `M` | 18.75 cm, 5 cm | LiDAR-to-bumper offset used by the model, safety margin |
| `SAFE_THRESHOLD` | 75 cm | S in `FIXED` mode |
| `P_TARGET`, `N_MAX` | 0.05, 20000 | target crash probability, cap on N |
| `PSI_WINDOW`, `PSI_QUANTILE`, `PSI_MIN_SAMPLES` | 50, 0.9, 10 | ψ estimate |
| `SCAN_TIMEOUT_NS` | 0.5 s | stale-scan brake |
| `CALIBRATION_RUN` | True | run 0 before the measured runs |
| `RANDOM_START`, `START_MARGIN_CM` | True, 3 cm | random start position |
| `ALIGN_*`, `RETURN_*` | | alignment and return controllers |

Module-level toggles at the top of `controller.py`: `DEFAULT_MODE`, `DEFAULT_RUNS`,
`ENABLE_CSV`, `usePingTauRX`, `CSV_DIR`.

---

## Troubleshooting

- **`PackageNotFoundError: No package metadata was found for monitor-pkg`**: the terminal
  was sourced before a `--symlink-install` rebuild. Open a new terminal and source
  `/opt/ros/jazzy/setup.bash` and `install/setup.bash` again.
- **New CSV columns or features are missing**: the copy in `install/` is out of date.
  Rebuild (`colcon build --symlink-install`), source again, and restart the nodes. The
  startup line and the CSV header show which version is running.
- **Changes to `client.py` have no effect**: restart `bot_monitor` on the bot.
- **`no d_brake from the bot`**: `bot_monitor` is not running, or the bot did not reach a
  standstill within 3 s.
- **`calibration failed`**: run 0's cruise speed or d_brake was missing or too far from the
  constants. Check that the start is far enough from S to reach cruise speed and that
  `bot_monitor` is running, or update the constants.
- **The bot turns away from the wall while aligning**: set `ALIGN_SIGN = -1`.
