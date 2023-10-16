#!/bin/bash

#####################################################
# Name of the image and container
#####################################################
readonly name=test

#####################################################
# src_data is the address at the host
# dest_data is the address at the container
# This folder is READ ONLY hence, algorithms run inside
# the container can read the data from the folder but
# they cannot save data in it
#####################################################
readonly src_data=$PWD/data/
readonly dest_data=/program/data/:ro

#####################################################
# src_results is the address at the host
# dest_results is the address at the container
# Algorithms run inside the container
# can read and write the data from the folder
# Data saved outside the designated directory will
# be erased when the container is stopped
#####################################################
readonly src_results=$PWD/experiments/results/
readonly dest_results=/program/experiments/results

#####################################################
# Location of the executable inside the container
#####################################################
readonly executable=/program/experiments/runner.py

#####################################################
# Stop running container
#####################################################
docker stop $name

#####################################################
# 1. Build the container (in case source files were updated)
#####################################################
docker build --tag $name .

#####################################################
# Clean files
#####################################################
docker system prune --force

#####################################################
# Automatically determine if GPUs are available
#####################################################
gpus=$([ $(ls -la /dev | grep nvidia | wc -l) "==" "0" ] && echo "" || echo "--gpus all")

#####################################################
# Start container
# Mount a directory data directory as read-only
# Mount results directory.
#####################################################
docker run $gpus -d --name $name -v $src_results:$dest_results -v $src_data:$dest_data -it $name /bin/bash

#####################################################
# Execute the algorithm in the container 
# and pass all arguments passed to this script

#####################################################
docker exec -it $name python $executable "$@"

#####################################################
# Stop running container
#####################################################
docker stop $name