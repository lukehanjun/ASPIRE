#!/usr/bin/env bash
# Run in a persistent terminal, or detach with setsid/nohup as in docs/libero-a40.md.
set -euo pipefail
ASPIRE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ASPIRE_ROOT/scripts/activate_libero.sh"
cd "$ASPIRE_ROOT"
exec scripts/supervise_services.sh --profile "$CAP_HARNESS_PROFILE" \
  --providers sam3,contact_graspnet,pyroki,curobo "$@"
