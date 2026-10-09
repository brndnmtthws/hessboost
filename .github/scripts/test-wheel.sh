#!/bin/sh
# Test prebuilt wheels against the Python test suite. Run from python/.
#
# Usage: test-wheel.sh PYTHON WHEEL_GLOBS
#
# WHEEL_GLOBS is hessboost's wheel, optionally followed by
# hessboost-runtime-cuda's (space-separated globs, one wheel each).
# Recreates the project environment for PYTHON with only the locked test
# group, installs the wheels into it (`--no-sync` keeps uv from building the
# project) and checks that the tests import the installed package, not the
# sources in the checkout. On a free-threaded interpreter, importing
# hessboost must leave the GIL disabled.
set -eu

python="$1"
wheel_glob="$2"

# scikit-learn publishes no musllinux wheels, and building it from source
# for every interpreter would dominate the job. On musl, test without it and
# without the tests that import it (test_polars.py's estimator tests skip
# themselves); the estimators are pure Python over the same extension the
# other tests exercise.
sync_args=
pytest_args=
if ldd --version 2>&1 | grep -q musl; then
  sync_args="--no-install-package scikit-learn"
  pytest_args="--ignore=tests/test_sklearn.py --deselect=tests/test_training.py::test_cv_accepts_explicit_folds_and_splitters"
fi
# polars publishes only abi3 wheels, which free-threaded interpreters cannot
# load (and building it from source would dominate the job), so test
# without it and its tests there; its conversion is pure Python.
if uv run --no-project --python "$python" python -c "import sys, sysconfig; sys.exit(not sysconfig.get_config_var('Py_GIL_DISABLED'))"; then
  sync_args="$sync_args --no-install-package polars --no-install-package polars-runtime-32"
  pytest_args="$pytest_args --ignore=tests/test_polars.py"
fi

# The argument lists and the globs are unquoted on purpose: the lists split
# into their arguments, and each glob expands to its matching wheel (a glob
# without a match stays literal and uv reports the missing file).
# shellcheck disable=SC2086
uv sync --locked --only-group test --python "$python" $sync_args
# shellcheck disable=SC2086
uv pip install $wheel_glob
uv run --no-sync python -c "import hessboost; assert 'site-packages' in hessboost.__file__, hessboost.__file__"
uv run --no-sync python -c "import sys, sysconfig, hessboost; assert not (sysconfig.get_config_var('Py_GIL_DISABLED') and sys._is_gil_enabled())"
# shellcheck disable=SC2086
uv run --no-sync pytest -p no:cacheprovider tests $pytest_args
