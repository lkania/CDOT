#!/bin/bash

# parse parameters
server=${server:-hydra12}
method=${method:-bin_mle}
k=${k:-5}
std_signal_region=${std_signal_region:-3}
data_id=${data_id:-50}
session=${session:-olaf}
lambda=${lambda:-0.01}
mu=${mu:-450}
sigma=${sigma:-20}
rate=${rate:-0.003}
a=${a:-250}
b=${b:-0}

while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        param="${1/--/}"
        declare $param="$2"
   fi

  shift
done

co="cd remote/olaf-analysis-np/experiments/; chmod +x run.sh; ./run.sh "

co+="--method $method "
co+="--k $k "

co+="--a $a "
co+="--rate $rate "
co+="--b $b "

co+="--std_signal_region $std_signal_region "
co+="--data_id $data_id "

co+="--lambda $lambda "
co+="--mu $mu "
co+="--sigma $sigma "

co+="--session $session ; exit"
ssh $server -t "$co"


