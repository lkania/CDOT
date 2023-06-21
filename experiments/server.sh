#!/bin/bash

# parse parameters
user=${user:-lkania}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

# load functions
. panes.sh

# load servers
. servers.sh

tmux kill-session -t monitor
tmux new -s monitor -d
10panes

tmux set -g pane-border-status top

COUNTER=0
for server in "${servers[@]}"; do
  tmux send-keys -t monitor.$COUNTER "ssh -t $server htop --user $user" ENTER
  tmux select-pane -t $COUNTER -T "$server"
  COUNTER=$((COUNTER+1))
done

#tmux set -g pane-border-format '#(sleep 0.5; ps -t #{pane_tty} -o args= | tail -n 1)'
tmux attach -t monitor