---
name: Lightweight mode and pipeline status
description: M1 8GB OOM issue resolved with LIGHTWEIGHT_MODE; Phase 2 labeling complete on 200K-row subset
type: project
---

Full 2.2M row dataset causes exit code 137 (OOM) on MacBook Air M1 8GB.

**Fix applied 2026-04-09:**
- `config.py`: Added LIGHTWEIGHT_MODE=True, SAMPLE_ROWS=200_000, USE_FLOAT32=True
- `src/data_prep.py`: load_raw_data() loads last 200K rows when LIGHTWEIGHT_MODE=True; converts float64→float32
- `src/cusum.py`: gc.collect() after filter completes
- `src/labeling.py`: gc.collect() after each TBM run
- `src/pipeline_runner.py`: sequential phase runner (A→B→C) with del + gc.collect() between phases

**Phase 2 results (200K-row lightweight run):**
- df_clean: 199,999 bars (2020-03-12 to 2025-03-31 range, last 200K)
- CUSUM events: 33,309 (16.66% event rate at m=2.0)
- Standard TBM: 33,297 labeled | crash(1)=18,750 (56.31%), no-crash(0)=14,547 (43.69%)
- Adaptive TBM: 33,297 labeled | crash(1)=18,714 (56.20%), no-crash(0)=14,583 (43.80%)
- Label agreement between schemes: 93.9%
- All outputs saved to data/processed/ (~6.7MB total)

**Why:** LIGHTWEIGHT_MODE=False needed for final paper results on HPC. Use pipeline_runner.py as the entry point.

**How to apply:** Always check config.LIGHTWEIGHT_MODE before suggesting full-data runs. When results look off (e.g. 56% crash rate is high), note it may be a subset artifact — full dataset had 15.36% event rate but crash rate within events was not yet computed.
