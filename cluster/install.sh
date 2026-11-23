#!/usr/bin/env bash
# One-time setup of the Claude gateway on a SLURM cluster, using YOUR normal SSH access.
#
#   cluster/install.sh hyperion                                   # everything in ~/carsim-claude
#   cluster/install.sh hyperion --base /scratch/me/carsim-claude  # on scratch (home over quota)
#   cluster/install.sh hyperion --module Python/3.11.3-GCCcore-12.3.0   # python from a module
#   cluster/install.sh hyperion --no-venv                         # skip the Python venv
#   cluster/install.sh hyperion --module Miniforge3 --python /scratch/me/envs/carsim_venv/bin/python
#                                                   # use a venv you already have (implies --no-venv)
#   cluster/install.sh hyperion --key-type rsa      # if the cluster does not accept ed25519 user keys
#
# 1. makes a new key ~/.ssh/claude_carsim_<alias> just for the gateway (ed25519, no passphrase:
#    it can only ever run the gateway),
# 2. installs cluster/gateway.sh as <base>/bin/gateway and a gateway.conf (if none yet),
# 3. creates <base>/venv with numpy and scipy (unless --no-venv),
# 4. adds the key to ~/.ssh/authorized_keys on the cluster as
#        restrict,command="CARSIM_GW_BASE=<base> <base>/bin/gateway" ssh-ed25519 ... claude-carsim
# 5. writes cluster/cluster.env here and checks that `cluster/ctl info` works.
# Re-running is safe: existing key, conf, venv and authorized_keys line are kept.
# Undo: delete that authorized_keys line (and <base> if you want).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: cluster/install.sh <ssh alias or user@host> [--base DIR] [--module NAME] [--no-venv]}"
shift
BASE='$HOME/carsim-claude'; MODULE=""; VENV=1; PY=""; KTYPE="ed25519"
while (( $# )); do
  case "$1" in
    --base) BASE="${2:?--base needs a directory}"; shift 2 ;;
    --module) MODULE="${2:?--module needs a module name}"; shift 2 ;;
    --no-venv) VENV=0; shift ;;
    --key-type) KTYPE="${2:?--key-type ed25519|rsa}"; shift 2 ;;
    --python) PY="${2:?--python needs the path of a python binary}"; VENV=0; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[[ "$BASE" =~ ^[A-Za-z0-9_./\$-]+$ ]] || { echo "--base: plain path please" >&2; exit 2; }
[[ -z "$MODULE" || "$MODULE" =~ ^[A-Za-z0-9_./+-]+$ ]] || { echo "--module: bad name" >&2; exit 2; }
[[ -z "$PY" || "$PY" =~ ^[A-Za-z0-9_./\$-]+$ ]] || { echo "--python: plain path please" >&2; exit 2; }
alias_clean="$(echo "$TARGET" | tr -c 'A-Za-z0-9_-' '_')"
[[ "$KTYPE" == ed25519 || "$KTYPE" == rsa ]] || { echo "--key-type: ed25519 or rsa" >&2; exit 2; }
KEY="$HOME/.ssh/claude_carsim_${alias_clean%_}"
[[ "$KTYPE" == rsa ]] && KEY="${KEY}_rsa"

host="$(ssh -G "$TARGET" | awk '$1=="hostname"{print $2}')"
user="$(ssh -G "$TARGET" | awk '$1=="user"{print $2}')"
port="$(ssh -G "$TARGET" | awk '$1=="port"{print $2}')"
echo "cluster: $user@$host port $port"

if [[ ! -f "$KEY" ]]; then
  if [[ "$KTYPE" == rsa ]]; then ssh-keygen -q -t rsa -b 4096 -N "" -C "claude-carsim" -f "$KEY"
  else ssh-keygen -q -t ed25519 -N "" -C "claude-carsim" -f "$KEY"; fi
fi
pub="$(cat "$KEY.pub")"

# Shell lines that make the module's python available (in the installer and in every job).
LOAD=""
[[ -n "$MODULE" ]] && LOAD="source /etc/profile >/dev/null 2>&1 || true; module purge; module load $MODULE"

echo "== checking the cluster"
B="$(ssh "$TARGET" "echo $BASE")"   # the base with \$HOME expanded on the cluster
echo "base: $B"
[[ -n "$PY" ]] || PY="$B/venv/bin/python"
ssh "$TARGET" "for c in sbatch squeue sacct scancel git apptainer singularity; do printf '%-12s %s\n' \$c \"\$(command -v \$c || echo -)\"; done
  $LOAD
  python3 --version"

echo "== installing the gateway"
ssh "$TARGET" "mkdir -p '$B/bin' && chmod 750 '$B'"
ssh "$TARGET" "cat > '$B/bin/gateway.new' && chmod 500 '$B/bin/gateway.new' && mv -f '$B/bin/gateway.new' '$B/bin/gateway'" < "$HERE/gateway.sh"
ssh "$TARGET" "cat > '$B/carsim.def'" < "$HERE/carsim.def"
{ cat "$HERE/gateway.conf.example"
  echo
  echo "# set by install.sh"
  echo "PYTHON=\"$PY\""
  echo "SETUP_ENV=\"$LOAD\""
} | ssh "$TARGET" "[ -f '$B/gateway.conf' ] && cat >/dev/null || { cat > '$B/gateway.conf' && chmod 400 '$B/gateway.conf'; }"

if (( VENV )); then
  echo "== python venv (numpy, scipy)"
  ssh "$TARGET" "set -e; $LOAD
    python3 -c 'import sys; assert sys.version_info >= (3, 10), \"python \" + sys.version.split()[0] + \" is too old: carsim needs 3.10+ (use --module)\"'
    [ -x '$B/venv/bin/python' ] || python3 -m venv '$B/venv'
    '$B/venv/bin/pip' install -q --upgrade pip 'numpy>=1.24' 'scipy>=1.10'
    '$B/venv/bin/python' -c 'import sys,numpy,scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)'"
fi

echo "== checking the python the jobs will use: $PY"
ssh "$TARGET" "$LOAD
  '$PY' -c 'import sys, numpy, scipy; assert sys.version_info >= (3, 10), \"python \" + sys.version.split()[0] + \" is too old: carsim needs 3.10+\"; print(\"python\", sys.version.split()[0], \"numpy\", numpy.__version__, \"scipy\", scipy.__version__)'"

echo "== authorising the restricted key"
ssh "$TARGET" "set -e; mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys
  grep -qF '${pub#* }' ~/.ssh/authorized_keys || echo 'restrict,command=\"CARSIM_GW_BASE=$B $B/bin/gateway\" $pub' >> ~/.ssh/authorized_keys
  grep -F 'claude-carsim' ~/.ssh/authorized_keys"

cat > "$HERE/cluster.env" <<ENV
CLUSTER_HOST=$host
CLUSTER_USER=$user
CLUSTER_PORT=$port
CLUSTER_KEY=$KEY
ENV
echo "== test: cluster/ctl info"
"$HERE/ctl" info
echo "done. Edit $B/gateway.conf on the cluster for partition/account/limits (chmod u+w first)."
