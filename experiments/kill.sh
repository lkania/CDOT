#!/bin/bash

tmux kill-server
ssh -t hydra12 tmux kill-server
ssh -t hydra11 tmux kill-server
ssh -t hydra10 tmux kill-server
ssh -t hydra9 tmux kill-server
ssh -t hydra8 tmux kill-server
ssh -t hydra7 tmux kill-server
ssh -t hydra6 tmux kill-server
ssh -t hydra5 tmux kill-server
ssh -t hydra4 tmux kill-server
ssh -t hydra3 tmux kill-server
ssh -t hydra2 tmux kill-server
ssh -t hydra1 tmux kill-server

ssh -t hydra12 pkill -9 -f pydevconsole.py
ssh -t hydra11 pkill -9 -f pydevconsole.py
ssh -t hydra10 pkill -9 -f pydevconsole.py
ssh -t hydra9 pkill -9 -f pydevconsole.py
ssh -t hydra8 pkill -9 -f pydevconsole.py
ssh -t hydra7 pkill -9 -f pydevconsole.py
ssh -t hydra6 pkill -9 -f pydevconsole.py
ssh -t hydra5 pkill -9 -f pydevconsole.py
ssh -t hydra4 pkill -9 -f pydevconsole.py
ssh -t hydra3 pkill -9 -f pydevconsole.py
ssh -t hydra2 pkill -9 -f pydevconsole.py
ssh -t hydra1 pkill -9 -f pydevconsole.py