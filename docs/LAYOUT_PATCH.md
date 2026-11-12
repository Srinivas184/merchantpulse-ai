# V11 layout adjustment

The supplied one-time `fix_layout.py` adjustment has been applied to `app.py` in this release copy.

- The four primary tabs now sit directly below the three “Start here” cards.
- The KPI row and “Today's briefing” block now render inside the Dashboard tab.
- The original source ZIP and local project were not modified.
- The helper script itself is not included in the release repository because it is a one-time migration utility and creates an extra `.bak` file when run.

Verification after this adjustment:

- `python -m compileall -q .`: passed.
- 75 non-UI unit tests: passed.
- `python benchmark_detectors.py`: reproduced the shipped benchmark counts.
- The 3 Streamlit AppTest checks were not run in the isolated review environment because Streamlit was unavailable and package installation could not reach the package index. Run them locally with `python -m unittest -v test_app_ui` after installing `requirements.txt`.
