#!/usr/bin/env bash
# carsim cluster gateway: the ONLY program the Claude SSH key may run on the cluster.
#
# It is installed as a forced command in ~/.ssh/authorized_keys:
#   restrict,command="$HOME/carsim-claude/bin/gateway" ssh-ed25519 AAAA... claude-carsim
# so whatever the client asks for arrives in $SSH_ORIGINAL_COMMAND and is checked here against a
# short list of verbs. No shell, no port forwarding, no agent, no pty. Everything happens under
# $BASE (default ~/carsim-claude): its own code snapshots, runs, logs and an audit log. Your own
# checkouts and runs elsewhere on the cluster are never touched.
#
# Verbs:
#   help                              this list
#   info                              sinfo summary and the gateway's limits
#   status                            your queued/running claude-* jobs
#   queue                             queue length, estimated start of pending jobs, and of 16/32/64-CPU jobs
#   history [days]                    claude-* jobs of the last days (sacct), default 3
#   push-code                         stdin: a git bundle; its branches go into $BASE/repo.git
#   submit <run> <sha> <cpus> <hours> [train flags...]
#                                     sbatch: python -m carsim.learn train --name <run> ... from
#                                     commit <sha>, --workers <cpus>, --hours <hours>
#   cancel <jobid>                    scancel, only for claude-* jobs
#   log <jobid> [lines]               tail of the job's output
#   runs                              every run with its latest progress line
#   progress <run> [lines]            tail of the run's progress.csv
#   get-run <run>                     stdout: tar of the run's policy, config, progress (+ checkpoint)
#   put-run <run>                     stdin: tar of policy.npy/config.json/progress.csv/checkpoint.npz
#                                     (only into a run that does not exist yet, e.g. to --init-from)
set -euo pipefail
umask 027
export LC_ALL=C

BASE="${CARSIM_GW_BASE:-$HOME/carsim-claude}"
CONF="$BASE/gateway.conf"
# Defaults; gateway.conf (written by install.sh, edit it on the cluster) overrides them.
PYTHON="$BASE/venv/bin/python"
SETUP_ENV=""            # shell lines run at the start of every job, e.g. "module load Python/3.11"
PARTITION=""
ACCOUNT=""
QOS=""
MAX_CPUS=64
MAX_HOURS=48
MAX_JOBS=4              # claude-* jobs queued or running at once
MEM_PER_CPU="1G"
CONTAINER=""            # path to a .sif image (cluster/carsim.def): jobs then run in Apptainer and see
                        # only their code snapshot (read-only) and $BASE/ai_runs, not the rest of $HOME
CONTAINER_CMD="apptainer"   # or "singularity"
# shellcheck source=/dev/null
[[ -f "$CONF" ]] && source "$CONF"

mkdir -p "$BASE"/{logs,ai_runs,code,tmp}
AUDIT="$BASE/audit.log"
RAW="${SSH_ORIGINAL_COMMAND:-help}"
client="${SSH_CLIENT:-local}"; printf '%s %s %s\n' "$(date -Is)" "${client%% *}" "$RAW" >> "$AUDIT"

die() { echo "gateway: $*" >&2; exit 2; }
# Split on whitespace only: no quotes, no globbing, no expansion of what the client sent.
read -r -a ARGV <<< "$RAW" || true
VERB="${ARGV[0]:-help}"
ARGS=("${ARGV[@]:1}")

RE_RUN='^[a-z0-9][a-z0-9_-]{0,47}$'
RE_SHA='^[0-9a-f]{7,40}$'
RE_INT='^[0-9]{1,6}$'
RE_JOB='^[0-9]{1,12}$'
RE_NUM='^[0-9]{1,4}(\.[0-9]{1,4})?$'
RE_TRACK='^(short|test|hill|monza|random|random:[0-9]{1,10}|power:[0-9]{1,10})$'
RE_TRACKS='^[a-z0-9:]{1,24}(,[a-z0-9:]{1,24}){0,7}$'

need() { [[ "$2" =~ $3 ]] || die "bad $1: '$2'"; }
lines_arg() { local n="${1:-40}"; need lines "$n" "$RE_INT"; (( n <= 2000 )) || n=2000; echo "$n"; }

claude_jobs() {  # "jobid name state elapsed timelimit reason" for this user's claude-* jobs
  squeue -u "$USER" -h -o '%i %j %T %M %l %R' | awk '$2 ~ /^claude-/' || true
}

case "$VERB" in
help)
  sed -n '/^# Verbs:/,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//' ;;

info)
  sinfo -s || true
  echo "limits: MAX_CPUS=$MAX_CPUS MAX_HOURS=$MAX_HOURS MAX_JOBS=$MAX_JOBS partition=${PARTITION:-default} account=${ACCOUNT:-default} qos=${QOS:-default}"
  ( set +eu; eval "$SETUP_ENV" >/dev/null 2>&1
    "$PYTHON" -c 'import sys, numpy, scipy; print("python", sys.version.split()[0], "numpy", numpy.__version__, "scipy", scipy.__version__)' ) 2>&1 \
    || echo "python env not ready: $PYTHON" ;;

status)
  echo "JOBID NAME STATE ELAPSED LIMIT REASON"; claude_jobs ;;

queue)  # read-only: where the claude-* jobs stand in the queue, and what a smaller job would get
  part="${PARTITION:-$(sinfo -h -o '%P' | awk '/\*$/ {sub(/\*$/, ""); print; exit}')}"
  echo "partition $part: $(squeue -p "$part" -t PD -h | wc -l) jobs pending, $(squeue -p "$part" -t R -h | wc -l) running"
  echo "-- estimated start (SLURM's guess, changes as jobs finish early)"
  claude_jobs | awk '$3 == "PENDING" {print $1}' | while read -r j; do
    squeue -h -j "$j" --start -o '%i %j %S %r' 2>/dev/null || true
    rank="$(squeue -p "$part" -t PD -h --sort=-p -o '%i' | grep -n "^$j\$" | cut -d: -f1 || true)"
    echo "   $j is number ${rank:-?} by priority among pending jobs"
  done
  for c in 16 32 64; do
    t="$(sbatch --test-only -p "$part" -N 1 -c "$c" -t 23:30:00 --wrap=true 2>&1 | sed -n 's/.*to start at \([^ ]*\).*/\1/p')"
    echo "   a new ${c}-CPU 23.5 h job would start: ${t:-unknown}"
  done ;;

history)
  d="${ARGS[0]:-3}"; need days "$d" "$RE_INT"
  sacct -u "$USER" -X -S "now-${d}days" -n -P -o JobID,JobName,State,Elapsed,Start,End,ExitCode \
    | awk -F'|' '$2 ~ /^claude-/' ;;

push-code)
  command -v git >/dev/null || die "git not found on the cluster"
  [[ -d "$BASE/repo.git" ]] || git init -q --bare "$BASE/repo.git"
  tmp="$(mktemp "$BASE/tmp/bundle.XXXXXX")"; trap 'rm -f "$tmp"' EXIT
  head -c 104857600 > "$tmp"  # 100 MB cap
  git --git-dir="$BASE/repo.git" bundle verify -q "$tmp" >/dev/null 2>&1 || die "not a valid git bundle"
  git --git-dir="$BASE/repo.git" fetch -q "$tmp" '+refs/heads/*:refs/heads/*'
  git --git-dir="$BASE/repo.git" for-each-ref --format='%(objectname:short) %(refname:short)' refs/heads ;;

submit)
  (( ${#ARGS[@]} >= 4 )) || die "usage: submit <run> <sha> <cpus> <hours> [train flags...]"
  run="${ARGS[0]}"; sha="${ARGS[1]}"; cpus="${ARGS[2]}"; hours="${ARGS[3]}"
  need run "$run" "$RE_RUN"; need sha "$sha" "$RE_SHA"; need cpus "$cpus" "$RE_INT"; need hours "$hours" "$RE_NUM"
  (( cpus >= 1 && cpus <= MAX_CPUS )) || die "cpus must be 1..$MAX_CPUS"
  awk -v h="$hours" -v m="$MAX_HOURS" 'BEGIN{exit !(h > 0 && h <= m)}' || die "hours must be >0 and <= $MAX_HOURS"
  n_jobs="$(claude_jobs | wc -l)"; (( n_jobs < MAX_JOBS )) || die "already $n_jobs claude jobs (MAX_JOBS=$MAX_JOBS)"
  claude_jobs | awk -v j="claude-$run" '$2 == j {found=1} END{exit !found}' && die "run $run already has a job"
  # Training flags: an allow-list of flags, each with the shape of value it may take.
  flags=(); rest=("${ARGS[@]:4}"); i=0
  while (( i < ${#rest[@]} )); do
    f="${rest[$i]}"; v="${rest[$((i+1))]:-}"
    case "$f" in
      --carry-over|--legacy-car) flags+=("$f"); i=$((i+1)); continue ;;
      --track|--test-track) need "$f" "$v" "$RE_TRACK" ;;
      --tracks|--practice-tracks) need "$f" "$v" "$RE_TRACKS" ;;
      --select) need "$f" "$v" '^(score|laps)$' ;;
      --init-from) need "$f" "$v" "$RE_RUN" ;;
      --population|--laps|--starts|--generations|--tracks-per-gen|--test-laps|--test-every|--seed) need "$f" "$v" "$RE_INT" ;;
      --power-share|--fast-corner-share|--steer-penalty|--max-time|--speed-bonus|--reversal-penalty|--limits-penalty|--steer-smoothing|--command-penalty|--pace-penalty|--edge-penalty|--assist-pace) need "$f" "$v" "$RE_NUM" ;;
      *) die "flag not allowed: $f" ;;
    esac
    flags+=("$f" "$v"); i=$((i+2))
  done
  # The code: a snapshot of exactly that commit (from push-code), shared by every job that uses it.
  full="$(git --git-dir="$BASE/repo.git" rev-parse --verify -q "$sha^{commit}")" || die "unknown commit $sha (push-code first)"
  code="$BASE/code/$full"
  if [[ ! -d "$code" ]]; then
    mkdir -p "$code.tmp" && git --git-dir="$BASE/repo.git" archive "$full" | tar -x -C "$code.tmp" && mv "$code.tmp" "$code"
  fi
  script="$BASE/tmp/claude-$run.sbatch"
  {
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=claude-$run"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=$cpus"
    echo "#SBATCH --mem-per-cpu=$MEM_PER_CPU"
    # wall time: the training stops itself after --hours (and saves every generation); +30 min margin
    echo "#SBATCH --time=$(awk -v h="$hours" 'BEGIN{m=int(h*60+30); printf "%d:%02d:00", m/60, m%60}')"
    echo "#SBATCH --output=$BASE/logs/%j.out"
    [[ -n "$PARTITION" ]] && echo "#SBATCH --partition=$PARTITION"
    [[ -n "$ACCOUNT" ]] && echo "#SBATCH --account=$ACCOUNT"
    [[ -n "$QOS" ]] && echo "#SBATCH --qos=$QOS"
    echo "set -euo pipefail"
    echo "set +eu  # module and profile scripts use unset variables"
    echo "$SETUP_ENV"
    echo "set -eu"
    echo "echo \"commit $full  run $run  host \$(hostname)  start \$(date -Is)\""
    if [[ -n "$CONTAINER" ]]; then
      for pre in APPTAINERENV SINGULARITYENV; do
        echo "export ${pre}_CARSIM_RUNS=/runs ${pre}_PYTHONPATH=/code ${pre}_OMP_NUM_THREADS=1 ${pre}_OPENBLAS_NUM_THREADS=1"
      done
      printf "exec %s exec --containall --cleanenv --bind '%s:/code:ro' --bind '%s:/runs' --pwd /code '%s' python -u" \
        "$CONTAINER_CMD" "$code" "$BASE/ai_runs" "$CONTAINER"
    else
      echo "cd '$code'"
      echo "export CARSIM_RUNS='$BASE/ai_runs' PYTHONPATH='$code' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
      printf "exec '%s' -u" "$PYTHON"
    fi
    printf " -m carsim.learn train --name %s --workers %s --hours %s" "$run" "$cpus" "$hours"
    printf " %s" "${flags[@]}"
    echo
  } > "$script"
  printf '%s run=%s sha=%s cpus=%s hours=%s flags=%s\n' "$(date -Is)" "$run" "$full" "$cpus" "$hours" "${flags[*]}" >> "$BASE/submissions.log"
  sbatch --parsable "$script" ;;

cancel)
  j="${ARGS[0]:-}"; need jobid "$j" "$RE_JOB"
  name="$(squeue -u "$USER" -h -j "$j" -o '%j' 2>/dev/null || true)"
  [[ "$name" == claude-* ]] || die "job $j is not a running/queued claude-* job of $USER"
  scancel "$j" && echo "cancelled $j ($name)" ;;

log)
  j="${ARGS[0]:-}"; need jobid "$j" "$RE_JOB"; n="$(lines_arg "${ARGS[1]:-}")"
  [[ -f "$BASE/logs/$j.out" ]] || die "no log for job $j"
  tail -n "$n" "$BASE/logs/$j.out" ;;

runs)
  shopt -s nullglob
  for d in "$BASE"/ai_runs/*/; do
    r="$(basename "$d")"; last="$(tail -n 1 "$d/progress.csv" 2>/dev/null || echo '-')"
    echo "$r  $last"
  done ;;

progress)
  r="${ARGS[0]:-}"; need run "$r" "$RE_RUN"; n="$(lines_arg "${ARGS[1]:-}")"
  f="$BASE/ai_runs/$r/progress.csv"; [[ -f "$f" ]] || die "no progress for run $r"
  head -n 1 "$f"; tail -n "$n" "$f" | sed '1{/^generation/d}' ;;

get-run)
  r="${ARGS[0]:-}"; need run "$r" "$RE_RUN"; d="$BASE/ai_runs/$r"; [[ -d "$d" ]] || die "no run $r"
  files=(); for f in policy.npy config.json progress.csv checkpoint.npz; do [[ -f "$d/$f" ]] && files+=("$f"); done
  tar -C "$d" -cf - "${files[@]}" ;;

put-run)
  r="${ARGS[0]:-}"; need run "$r" "$RE_RUN"; d="$BASE/ai_runs/$r"
  [[ -e "$d" ]] && die "run $r already exists on the cluster (refusing to overwrite)"
  tmp="$(mktemp "$BASE/tmp/run.XXXXXX")"; trap 'rm -f "$tmp"' EXIT
  head -c 209715200 > "$tmp"  # 200 MB cap
  # Only regular files with these exact names, nothing else (no paths, links or directories).
  tar -tvf "$tmp" | awk '{print substr($1,1,1), $NF}' | while read -r type name; do
    [[ "$type" == "-" ]] || die "put-run: '$name' is not a regular file"
    [[ "$name" =~ ^(\./)?(policy\.npy|config\.json|progress\.csv|checkpoint\.npz)$ ]] || die "put-run: file not allowed: $name"
  done
  mkdir -p "$d.tmp" && tar -xf "$tmp" -C "$d.tmp" --no-same-owner --no-same-permissions && mv "$d.tmp" "$d"
  ls -l "$d" ;;

*)
  die "unknown verb '$VERB' (try: help)" ;;
esac
