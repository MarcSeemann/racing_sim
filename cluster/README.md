# Training on a SLURM cluster through a restricted gateway

This lets an automated agent (Claude Code), or you, submit and watch training jobs on the cluster
**without a shell there**. It uses a dedicated SSH key that can only run `gateway.sh`, a short
list of checked commands.

```
this machine                                   cluster (e.g. hyperion)
cluster/ctl  ──ssh, restricted key only──▶  ~/.ssh/authorized_keys:
                                              restrict,command="~/carsim-claude/bin/gateway" <key>
                                            ~/carsim-claude/
                                              bin/gateway      the allow-list (gateway.sh)
                                              gateway.conf     limits, partition, python (only you edit)
                                              repo.git         code sent with push-code
                                              code/<sha>/      one read-only snapshot per commit
                                              ai_runs/<run>/   checkpoints (CARSIM_RUNS)
                                              logs/<jobid>.out audit.log submissions.log
```

## What the key can and cannot do

| Can | Cannot |
| --- | --- |
| `status`, `history`, `info`, `runs`, `progress`, `log` (read-only) | open a shell or a pty, forward ports or the agent, copy arbitrary files |
| `push-code`: send *committed* branches as a git bundle | touch anything outside `~/carsim-claude/` through the gateway |
| `submit`: `python -m carsim.learn train` from one exact commit, allow-listed flags only, capped CPUs, hours and number of jobs | run any other program, or pick another job name, output folder or `--workers` |
| `cancel` its own `claude-*` jobs | cancel your other jobs |
| `get-run` / `put-run` the four run files (`put-run` only into a run that doesn't exist yet) | overwrite an existing run, upload other files, links or paths |

Every call is appended to `~/carsim-claude/audit.log`, and every submission to `submissions.log`.

**The one thing to know:** a job runs the Python in the commit it was submitted from. The gateway
controls *which* program starts (carsim.learn, from a pinned commit you can inspect in
`code/<sha>/`), but that code runs as your cluster user. For a real sandbox, build the Apptainer
image (below). Jobs then run with `--containall` and see only their read-only code snapshot and
`ai_runs/`, not the rest of your home directory.

## Install (once, with your normal SSH access)

    cluster/install.sh hyperion

It makes `~/.ssh/claude_carsim_hyperion`, installs the gateway and a venv with numpy and scipy, adds
the restricted key to `authorized_keys`, writes `cluster/cluster.env` (git-ignored) and runs
`cluster/ctl info`. Then, on the cluster, set the partition/account and limits:

    chmod u+w ~/carsim-claude/gateway.conf && nano ~/carsim-claude/gateway.conf && chmod 400 ~/carsim-claude/gateway.conf

Optional container isolation:

    cd ~/carsim-claude && apptainer build carsim.sif carsim.def    # then CONTAINER="$HOME/carsim-claude/carsim.sif"

Undo everything: delete the `claude-carsim` line from `~/.ssh/authorized_keys` on the cluster.

## Use

    cluster/ctl push-code                                   # after committing
    cluster/ctl put-run random                              # upload ai_runs/random to start from it
    cluster/ctl submit monza-v2 64 24 --track random --init-from random --population 120 \
        --starts 8 --tracks-per-gen 4 --power-share 0.4 --fast-corner-share 0.25 --steer-penalty 2
    cluster/ctl status
    cluster/ctl log <jobid> 20
    cluster/ctl progress monza-v2 10
    cluster/ctl get-run monza-v2                            # into cluster_runs/monza-v2/ (not ai_runs/)
    cluster/ctl cancel <jobid>

A job stops itself after `<hours>` (SLURM gets 30 min of slack). Training saves every generation,
so submitting the same run name again resumes it.
