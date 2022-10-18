#!/bin/bash

function send {
  ssh $1 -t "cd remote/olaf-analysis-np/experiments/; chmod +x run.sh; ./run.sh --method ${2} --nnls ${3} --k ${4} --std_signal_region ${5} --no_signal ${6} --data_id ${7}; exit"
}

