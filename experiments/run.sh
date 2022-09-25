#!/bin/bash

# parse parameters
method=${method:-bin_mle}
k=${k:-30}
std_signal_region=${std_signal_region:-3}
no_signal=${no_signal:-False}
nnls=${nnls:-None}
data_id=${data_id:-50}

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

prefix="source activate olaf-analysis-np && python runner.py"
restart
tmux new -s olaf -d
tmux send-keys -t olaf.0 "${prefix} --method ${method} --nnls ${nnls} --k ${k} --std_signal_region ${std_signal_region} --no_signal ${no_signal} --data_id ${data_id}" ENTER
#tmux attach -t olaf

#function nnls {
#  b=$1
#  tmux send-keys -t olaf.1 "${a} mom -nnls ${b}" ENTER
#  tmux send-keys -t olaf.2 "${a} bin_chi2 -nnls ${b}" ENTER
#  tmux send-keys -t olaf.3 "${a} bin_mom -nnls ${b}" ENTER
#}

#case $1 in
#
#  #### working methods up to now
#
#  bin_mle)
#    tmux send-keys -t olaf.0 "${a} bin_mle -k ${2} -std_signal_region ${3}" ENTER
#    ;;
#
#  bin_mom_lawson_scipy)
#    nnls="-nnls lawson_scipy"
#    tmux send-keys -t olaf.1 "${a} bin_mom ${2} ${3} ${nnls}" ENTER
#    ;;
#
#  ####
#
#  lawson_scipy_1)
#    2panes
#    b="-nnls lawson_scipy"
#    tmux send-keys -t olaf.0 "${a} mom ${b}" ENTER
#    tmux send-keys -t olaf.1 "${a} bin_mom ${b}" ENTER
#    ;;
#
#  lawson_scipy_2)
#    b="-nnls lawson_scipy"
#    tmux send-keys -t olaf.0 "${a} bin_chi2 ${b}" ENTER
#    ;;
#
#  ####
#
#  pg_jax_1)
#    2panes
#    b="-nnls pg_jax"
#    tmux send-keys -t olaf.0 "${a} mom ${b}" ENTER
#    tmux send-keys -t olaf.1 "${a} bin_chi2 ${b}" ENTER
#    ;;
#
#  pg_jax_2)
#    b="-nnls pg_jax"
#    tmux send-keys -t olaf.0 "${a} bin_mom ${b}" ENTER
#    ;;
#
#  ####
#
#  lawson_jax_1)
#    2panes
#    b="-nnls lawson_jax"
#    tmux send-keys -t olaf.0 "${a} mom ${b}" ENTER
#    tmux send-keys -t olaf.1 "${a} bin_mom ${b}" ENTER
#    ;;
#
#  lawson_jax_2)
#    b="-nnls lawson_jax"
#    tmux send-keys -t olaf.0 "${a} bin_chi2 ${b}" ENTER
#    ;;
#
#  pg_jax)
#    4panes
#    nnls "pg_jax"
#    ;;
#
#  lawson_scipy)
#    4panes
#    nnls "lawson_scipy"
#    ;;
#
#  lawson_jax)
#    4panes
#    nnls "lawson_jax"
#    ;;
#
#  pg_jaxopt)
#    4panes
#    nnls "pg_jaxopt"
#    ;;
#
#  conic_cvx)
#    4panes
#    nnls "conic_cvx"
#    ;;
#
#  all)
#    10panes
#    tmux send-keys -t olaf.0 "${a} bin_mle" ENTER
#
#    b="-nnls lawson_scipy"
#    tmux send-keys -t olaf.1 "${a} mom ${b}" ENTER
#    tmux send-keys -t olaf.2 "${a} bin_chi2 ${b}" ENTER
#    tmux send-keys -t olaf.3 "${a} bin_mom ${b}" ENTER
#
#    b="-nnls lawson_jax"
#    tmux send-keys -t olaf.4 "${a} mom ${b}" ENTER
#    tmux send-keys -t olaf.5 "${a} bin_chi2 ${b}" ENTER
#    tmux send-keys -t olaf.6 "${a} bin_mom ${b}" ENTER
#
#    b="-nnls pg_jax"
#    tmux send-keys -t olaf.7 "${a} mom ${b}" ENTER
#    tmux send-keys -t olaf.8 "${a} bin_chi2 ${b}" ENTER
#    tmux send-keys -t olaf.9 "${a} bin_mom ${b}" ENTER
#    ;;
#esac