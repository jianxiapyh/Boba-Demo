#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Boba Phone Demo reuses the working phystwin environment from Boba-Batched."
echo "Running the phone-only dependency installer instead of modifying core Boba packages."
exec bash "${SCRIPT_DIR}/install_demo2_extras.sh" "$@"
