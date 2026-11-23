#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"
export PYTHONUTF8=1
exec python3 CAR_START.py "$@"
