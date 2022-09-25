#!/bin/bash

# load functions
. panes.sh

restart
tmux new -s monitor -d
10panes
tmux send-keys -t monitor.0 "ssh -t hydra12 htop --user lkania" ENTER
tmux send-keys -t monitor.1 "ssh -t hydra11 htop --user lkania" ENTER
tmux send-keys -t monitor.2 "ssh -t hydra10 htop --user lkania" ENTER
tmux send-keys -t monitor.3 "ssh -t hydra9 htop --user lkania" ENTER
tmux send-keys -t monitor.4 "ssh -t hydra8 htop --user lkania" ENTER
tmux send-keys -t monitor.5 "ssh -t hydra7 htop --user lkania" ENTER
tmux send-keys -t monitor.6 "ssh -t hydra6 htop --user lkania" ENTER
tmux send-keys -t monitor.7 "ssh -t hydra5 htop --user lkania" ENTER
tmux send-keys -t monitor.8 "ssh -t hydra4 htop --user lkania" ENTER
tmux send-keys -t monitor.9 "ssh -t hydra3 htop --user lkania" ENTER
tmux set -g pane-border-status top
tmux set -g pane-border-format '#(sleep 0.25; ps -t #{pane_tty} -o args= | tail -n 1)'
tmux attach -t monitor