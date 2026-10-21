# Changelog

## 2026-10-20: fixes found by a source read

These were found by reading the source against its own documentation after release, not by the earlier test and review passes. Each fix has a regression test in
`tests/test_findings_20261020.py` that failed before the fix, and each fix was checked by undoing it and confirming a test fails. Behaviour that changed is listed under "Changed".

### Fixed
- **A PRE revoked in a `PreDeck` did not stop training.** `TrainingRun(..., pre_deck=deck)` now refuses to start with a revoked PRE and halts a run in progress at its next step.
