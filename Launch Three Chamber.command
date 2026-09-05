#!/bin/bash
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo 'Set up the Python environment as described in README.md first.'
  exit 1
fi
exec .venv/bin/python scripts/launch.py
