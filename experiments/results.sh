#!/bin/bash

function f {

  inner_co="source activate olaf-analysis-np && python ./experiments/results.py --data_id $2"
  co+="tmux kill-session -t RESULTS; "
  co+="tmux new -s RESULTS -d; "
  co+="tmux send-keys -t RESULTS.0 \"$inner_co\" ENTER"
  ssh $1 -t "cd ./remote/olaf-analysis-np/; $co"

}

f hydra7 real
f hydra8 50

