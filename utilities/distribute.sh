#!/bin/bash

#############################################
# Save all parameters passed to distribute.sh
#############################################
parameters="$@"

#############################################
# Parse parameters
#############################################
. parser.sh
if [ -z ${server+x} ]; then echo "Server isn't specified"; exit; fi
if [ -z ${session+x} ]; then echo "Session isn't specified"; exit; fi
if [ -z ${rundocker+x} ]; then echo "Location of rundocker.sh isn't specified"; exit; fi

#############################################
# Declare the command to be run inside tmux session
#############################################
# Due to some Docker constraints, it is important to
# set the current working directory to $rundocker
# befoer executing rundocker.sh
# In the following command ./rundocker.sh $parameters means that
# all the parameters passed to distribute.sh
# will be forwarded to ./rundocker.sh
tcommand="cd $rundocker; chmod +x rundocker.sh; ./rundocker.sh $parameters"

#############################################
# If tmux isn't available, activate conda's base environment
#############################################
command="if ! type "tmux" > /dev/null 2>&1;
          then source ~/miniconda3/bin/activate base;"

#############################################
# If tmux isn't available in conda's base environment, install it
#############################################
command+="if ! type "tmux" > /dev/null 2>&1;
          then conda install -y -c conda-forge tmux;
          fi;
          fi;"

#############################################
# Connects to the server via ssh
#############################################
command+="tmux kill-session -t $session 2> /dev/null;"

#############################################
# Creates a tmux session
#############################################
command+="tmux new -s $session -d;"

#############################################
# Runs your command inside the tmux session
#############################################
command+="tmux send-keys -t $session.0 '$tcommand' ENTER;
          exit"

ssh $server -t "$command"




