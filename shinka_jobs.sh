#!/usr/bin/env bash
set -euo pipefail

# Optional arg: filter by results-dir fragment.
# Example:
#   ./shinka_jobs.sh results/results_circle_async_large
FILTER="${1:-results/}"

ps -axo pid=,etime=,pcpu=,pmem=,state=,command= |
awk -v filter="$FILTER" '
{
    pid=$1
    etime=$2
    pcpu=$3
    pmem=$4
    state=$5
    cmd=$0

    sub(/^[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]+/, "", cmd)
}

cmd ~ /(^|\/)(python|python3)([0-9.]*)[[:space:]]+evaluate\.py[[:space:]]+--program_path/ && index(cmd, filter) {

    gen="?"
    if (match(cmd, /gen_[0-9]+/)) {
        gen = substr(cmd, RSTART + 4, RLENGTH - 4)
    }

    printf "pid=%s gen=%s elapsed=%s cpu=%s%% mem=%s%% state=%s\n  %s\n\n", pid, gen, etime, pcpu, pmem, state, cmd
    found=1
}
END {
    if (!found) {
        print "No matching local Shinka eval jobs found."
    }
}'
