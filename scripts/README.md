# Parity fixtures

`gen_fixtures.py` trains **real XGBoost** on standardized synthetic datasets and
writes disjoint training and test datasets, parameters, and XGBoost test-set
predictions to `../fixtures/*.json`. The Rust test
`crates/sequoia-boost/tests/parity.rs` trains on the same training rows and
asserts that held-out RMSE or accuracy remains close to XGBoost.

```sh
pip install xgboost numpy
python scripts/gen_fixtures.py
cargo test -p sequoia-boost --test parity -- --ignored
```

Fixtures are intentionally not checked in and are regenerated in CI. The
datasets use `tree_method=hist` with a fixed `max_bin`. Pointwise prediction
differences are reported for diagnosis, while held-out model quality determines
pass or failure because independent histogram implementations need not select
identical split points.
