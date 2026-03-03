# Changelog

## [2.1.0] — 2026-03-03

### Fixed

#### `core/formation.py` — Adaptive sigma radius was non-functional
- **Bug**: `_build_with_sigma()` used the fixed `RADIUS_CAP` constant as the
  connectivity threshold regardless of the current `sigma` value. The adaptation
  loop therefore changed edge *weights* only — never edge *membership* — leaving
  disconnected clusters until the full kNN fallback fired on every normal graph.
- **Fix**: Replaced `RADIUS_CAP` with `sigma * 3.0` as the effective radius.
  Since `base_sigma = RADIUS_CAP / 3.0`, the initial pass is identical.
  Each sigma step (×1.5) now also expands the connectivity radius proportionally,
  allowing the loop to bridge real gaps (e.g. a GK isolated at x=8 m from the
  defensive line at x=18 m) without resorting to fallback.
- **Impact**: `test_nominal_graph_connected_sigma_one` now passes (`sigma×1.50`,
  `fallback_knn_only=false`). `test_spread_formation_adapts_sigma` also passes.

#### `tests/test_failure_modes.py` — FP stub missing `to_dict()`
- **Bug**: The lightweight `FP` stub created by `make_players()` had no
  `to_dict()` method. `pipeline.to_json()` calls `p.to_dict()` on every player,
  causing `AttributeError` in all four `TestSchemaVersion` tests.
- **Fix**: Added `to_dict()` to the `FP` stub returning serialisation-safe
  defaults for `pixel_pos`, `bbox`, and `confidence`.

#### `core/pipeline.py` — `process_positions()` method missing
- **Bug**: `demo.py --mode positions` calls `pipeline.process_positions(...)` but
  no such method existed on `FootballIntelligencePipeline`.
- **Fix**: Added `process_positions(home_positions, away_positions, ...)` that
  wraps raw pitch-coordinate lists into `Player` objects and runs the full
  formation + counter pipeline, bypassing detection and calibration.
- **Bonus**: Added `processing_time_ms` property alias on `FrameAnalysis` (maps
  to `processing_ms`) to satisfy the demo's attribute access.

#### `demo.py` — Velocity units caused permanent non-settlement
- **Bug**: `_velocities()` returned values in **m/s** (multiplied by `fps=25`).
  `SettledStateDetector.SETTLE_VELOCITY_THR = 2.5` is calibrated in **m/frame**
  (same unit as test fixture `velocity=(0.3, 0.1)`). Peak demo velocity
  ≈ 6.9 m/s >> 2.5, so `is_settled` was always `False`, producing 0 counters.
- **Fix**: Removed the `fps` multiplier from `_velocities()`. Velocities now
  returned in m/frame (max ≈ 0.28 m/frame), well within the threshold.
- **Result**: Demo produces `is_settled=True` from frame 1, `stab=0.97`,
  `150/150 frames with counters`.

### Changed

#### `README.md` — Consistency pass
- Expected demo output block updated to match v2.1 output format.
- Test count corrected: **22 → 21** (header + code block).
- Positions-mode performance line updated: **≈ 0.3 s / 150 frames** (~1.5 ms/frame).
