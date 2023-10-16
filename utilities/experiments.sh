#!/bin/bash

#############################################
# load servers
#############################################
. servers.sh

#############################################
# Declare parameters that change from server to server
#############################################
parameters=(
  20
  15
  10
  5
)

#############################################
# We iterate over the servers and run the script
# with different parameters in each one of them
#############################################
for i in "${!parameters[@]}"; do
  ./distribute.sh --server "${servers[i]}"  \
                  --session PARAMETERS      \
                  --rundocker "~/remote/olaf-analysis-np" \
                  --k "${parameters[i]}"    \
                  --method bin_mle          \
                  --bins 100                \
                  --std_signal_region 3     \
                  --folds 500               \
                  --data_id 50              \
                  --mu 450                  \
                  --sigma 20                \
                  --lambda 0.01             \
                  --rate 0.003              \
                  --a 250                   \
                  --b 0
done

