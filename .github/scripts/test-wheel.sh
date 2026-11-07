#!/bin/sh
# Test prebuilt wheels against the Python test suite. Run from python/.
#
# Usage: test-wheel.sh PYTHON WHEEL_GLOBS
#
# PYTHON is a CPython version, with a `t` suffix for the free-threaded build.
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

# Without the suffix, the GIL-enabled build: uv takes an installed
# free-threaded 3.14+ for a plain "3.14" (and keeps an environment of one),
# so the request says "+gil". uv downloads only the plain version, which is
# the GIL-enabled build, so install it first.
case "$python" in
  *t) request="$python" ;;
  *) request="$python+gil" ;;
esac
uv python install --no-bin "$python"

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
if uv run --no-project --python "$request" python -c "import sys, sysconfig; sys.exit(not sysconfig.get_config_var('Py_GIL_DISABLED'))"; then
  sync_args="$sync_args --no-install-package polars --no-install-package polars-runtime-32"
  pytest_args="$pytest_args --ignore=tests/test_polars.py"
fi

# The argument lists and the globs are unquoted on purpose: the lists split
# into their arguments, and each glob expands to its matching wheel (a glob
# without a match stays literal and uv reports the missing file).
#
# uv resolves the whole locked test group (numpy, scipy, scikit-learn, polars,
# ...) for every interpreter on every platform. The release workflow runs that
# across a wide, uncached wheel matrix, so a transient files.pythonhosted.org
# connection error can fail one wheel job; because crates.io, PyPI and the
# GitHub release all depend on the entire wheel matrix, one such blip skips the
# whole release. uv retries a failed request a few times on its own, but a CDN
# blip can outlast that budget. Retry the whole sync a few times with a short
# backoff: uv's cache makes each retry cheap (only the wheel that failed to
# download is fetched again).
sync_with_retry() {
  attempt=1
  while [ "$attempt" -lt 3 ]; do
    # shellcheck disable=SC2086
    if uv sync --locked --only-group test --python "$request" $sync_args; then
      return 0
    fi
    echo "uv sync failed (attempt $attempt/3); retrying in 10s" >&2
    sleep 10
    attempt=$((attempt + 1))
  done
  # Final attempt: a failure propagates through set -eu and fails the job.
  # shellcheck disable=SC2086
  uv sync --locked --only-group test --python "$request" $sync_args
}
sync_with_retry
# shellcheck disable=SC2086
uv pip install $wheel_glob
uv run --no-sync python -c "import hessboost; assert 'site-packages' in hessboost.__file__, hessboost.__file__"
uv run --no-sync python -c "import sys, sysconfig, hessboost; assert not (sysconfig.get_config_var('Py_GIL_DISABLED') and sys._is_gil_enabled())"
# shellcheck disable=SC2086
uv run --no-sync pytest -p no:cacheprovider tests $pytest_args
