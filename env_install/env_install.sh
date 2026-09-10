#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Boba Phone Demo includes its Boba runtime and requires the phystwin-cu132 dependency environment."
echo "Running the phone-only dependency installer instead of modifying core Boba packages."
exec bash "${SCRIPT_DIR}/install_demo2_extras.sh" "$@"
