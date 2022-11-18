#!/bin/bash

# parse parameters
session=${session:-olaf}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
#        echo $1 $2 # print assigned parameters
   fi

  shift
done

# load functions
. panes.sh

tmux kill-session -t monitor
tmux new -s monitor -d
10panes
tmux send-keys -t monitor.0 "ssh -t hydra12 tmux attach -t $session" ENTER
tmux send-keys -t monitor.1 "ssh -t hydra11 tmux attach -t $session" ENTER
tmux send-keys -t monitor.2 "ssh -t hydra10 tmux attach -t $session" ENTER
tmux send-keys -t monitor.3 "ssh -t hydra9 tmux attach -t $session" ENTER
tmux send-keys -t monitor.4 "ssh -t hydra8 tmux attach -t $session" ENTER
tmux send-keys -t monitor.5 "ssh -t hydra7 tmux attach -t $session" ENTER
tmux send-keys -t monitor.6 "ssh -t hydra6 tmux attach -t $session" ENTER
tmux send-keys -t monitor.7 "ssh -t hydra5 tmux attach -t $session" ENTER
tmux send-keys -t monitor.8 "ssh -t hydra4 tmux attach -t $session" ENTER
tmux send-keys -t monitor.9 "ssh -t hydra3 tmux attach -t $session" ENTER
tmux set -g pane-border-status top
tmux set -g pane-border-format '#(sleep 0.3; ps -t #{pane_tty} -o args= | tail -n 1)'
tmux attach -t monitor