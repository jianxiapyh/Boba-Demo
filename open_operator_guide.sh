#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GUIDE_PATH="${REPO_ROOT}/IMMERSIVE_DEMO_OPERATOR_GUIDE.html"

if [[ ! -f "${GUIDE_PATH}" ]]; then
  echo "Operator guide not found: ${GUIDE_PATH}" >&2
  exit 1
fi

if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "${GUIDE_PATH}" >/dev/null 2>&1 &
elif command -v gio >/dev/null 2>&1; then
  gio open "${GUIDE_PATH}" >/dev/null 2>&1 &
else
  echo "Open this file in a browser: ${GUIDE_PATH}"
  exit 0
fi

echo "Opened the local Boba operator guide: ${GUIDE_PATH}"
