#/bin/bash
set -Eeo pipefail
. $HOME/.profile
# Print NODE_NAME if set
if [ -n "$NODE_NAME" ]; then
  echo "NODE_NAME: $NODE_NAME"
fi
echo "Working directory: $(pwd)"
echo "Running command: $@"
exec "$@"
