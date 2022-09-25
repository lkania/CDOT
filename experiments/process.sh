#!/bin/bash

# load functions
. panes.sh

restart
tmux new -s monitor -d
10panes
tmux send-keys -t monitor.0 "ssh -t hydra12 tmux attach" ENTER
tmux send-keys -t monitor.1 "ssh -t hydra11 tmux attach" ENTER
tmux send-keys -t monitor.2 "ssh -t hydra10 tmux attach" ENTER
tmux send-keys -t monitor.3 "ssh -t hydra9 tmux attach" ENTER
tmux send-keys -t monitor.4 "ssh -t hydra8 tmux attach" ENTER
tmux send-keys -t monitor.5 "ssh -t hydra7 tmux attach" ENTER
tmux send-keys -t monitor.6 "ssh -t hydra6 tmux attach" ENTER
tmux send-keys -t monitor.7 "ssh -t hydra5 tmux attach" ENTER
tmux send-keys -t monitor.8 "ssh -t hydra4 tmux attach" ENTER
tmux send-keys -t monitor.9 "ssh -t hydra3 tmux attach" ENTER
tmux set -g pane-border-status top
tmux set -g pane-border-format '#(sleep 0.25; ps -t #{pane_tty} -o args= | tail -n 1)'
tmux attach -t monitor