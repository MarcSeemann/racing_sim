#!/usr/bin/env bash
# One-time setup of the Claude gateway on a SLURM cluster, using YOUR normal SSH access.
#
#   cluster/install.sh hyperion            # an alias from ~/.ssh/config (or user@host)
#   cluster/install.sh hyperion --no-venv  # skip creating the Python venv on the cluster
#
# 1. makes a new key ~/.ssh/claude_carsim_<alias> just for the gateway (ed25519, no passphrase:
#    it can only ever run the gateway),
# 2. installs cluster/gateway.sh as ~/carsim-claude/bin/gateway and a gateway.conf (if none yet),
# 3. creates ~/carsim-claude/venv with numpy and scipy (unless --no-venv),
# 4. adds the key to ~/.ssh/authorized_keys on the cluster as
#        restrict,command="<home>/carsim-claude/bin/gateway" ssh-ed25519 ... claude-carsim
# 5. writes cluster/cluster.env here and checks that `cluster/ctl help` works.
# Undo: delete that authorized_keys line (and ~/carsim-claude/ if you want).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: cluster/install.sh <ssh alias or user@host> [--no-venv]}"
VENV=1; [[ "${2:-}" == "--no-venv" ]] && VENV=0
alias_clean="$(echo "$TARGET" | tr -c 'A-Za-z0-9_-' '_')"
KEY="$HOME/.ssh/claude_carsim_${alias_clean%_}"

host="$(ssh -G "$TARGET" | awk '$1=="hostname"{print $2}')"
user="$(ssh -G "$TARGET" | awk '$1=="user"{print $2}')"
port="$(ssh -G "$TARGET" | awk '$1=="port"{print $2}')"
echo "cluster: $user@$host port $port"

[[ -f "$KEY" ]] || ssh-keygen -q -t ed25519 -N "" -C "claude-carsim" -f "$KEY"
pub="$(cat "$KEY.pub")"

echo "== checking the cluster"
ssh "$TARGET" 'for c in sbatch squeue sacct scancel git python3 apptainer singularity; do printf "%-12s %s\n" "$c" "$(command -v $c || echo -)"; done; python3 --version'

echo "== installing the gateway"
ssh "$TARGET" 'mkdir -p ~/carsim-claude/bin && chmod 750 ~/carsim-claude'
ssh "$TARGET" 'cat > ~/carsim-claude/bin/gateway && chmod 500 ~/carsim-claude/bin/gateway' < "$HERE/gateway.sh"
ssh "$TARGET" 'cat > ~/carsim-claude/carsim.def' < "$HERE/carsim.def"
ssh "$TARGET" '[ -f ~/carsim-claude/gateway.conf ] || { cat > ~/carsim-claude/gateway.conf && chmod 400 ~/carsim-claude/gateway.conf; }' < "$HERE/gateway.conf.example"

if (( VENV )); then
  echo "== python venv (numpy, scipy)"
  ssh "$TARGET" 'set -e; [ -x ~/carsim-claude/venv/bin/python ] || python3 -m venv ~/carsim-claude/venv
                 ~/carsim-claude/venv/bin/pip install -q --upgrade pip "numpy>=1.24" "scipy>=1.10"
                 ~/carsim-claude/venv/bin/python -c "import sys,numpy,scipy; print(sys.version.split()[0], numpy.__version__, scipy.__version__)"'
fi

echo "== authorising the restricted key"
ssh "$TARGET" "set -e; mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys
  grep -qF '${pub#* }' ~/.ssh/authorized_keys || echo \"restrict,command=\\\"\$HOME/carsim-claude/bin/gateway\\\" $pub\" >> ~/.ssh/authorized_keys
  grep -F 'claude-carsim' ~/.ssh/authorized_keys"

cat > "$HERE/cluster.env" <<ENV
CLUSTER_HOST=$host
CLUSTER_USER=$user
CLUSTER_PORT=$port
CLUSTER_KEY=$KEY
ENV
echo "== test: cluster/ctl info"
"$HERE/ctl" info
echo "done. Edit ~/carsim-claude/gateway.conf on the cluster for partition/account/limits (chmod u+w first)."
