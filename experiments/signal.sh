#!/bin/bash

./distribute.sh --server hydra12 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SIGNAL --lambda 0.01
./distribute.sh --server hydra11 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SIGNAL --lambda 0.005
./distribute.sh --server hydra10 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SIGNAL --lambda 0.001
./distribute.sh --server hydra9 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SIGNAL --lambda 0.0005
./distribute.sh --server hydra8 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SIGNAL --lambda 0.0001
./distribute.sh --server hydra7 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SIGNAL --lambda 0

./distribute.sh --server hydra6 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session SIGNAL_REAL --mu 45 --sigma 2 --a 30 --b 60 --lambda 0.01 --rate 0.01
./distribute.sh --server hydra5 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session SIGNAL_REAL --mu 45 --sigma 2 --a 30 --b 60 --lambda 0.005 --rate 0.01
./distribute.sh --server hydra4 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session SIGNAL_REAL --mu 45 --sigma 2 --a 30 --b 60 --lambda 0.001 --rate 0.01
./distribute.sh --server hydra3 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session SIGNAL_REAL --mu 45 --sigma 2 --a 30 --b 60 --lambda 0.0005 --rate 0.01
./distribute.sh --server hydra2 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session SIGNAL_REAL --mu 45 --sigma 2 --a 30 --b 60 --lambda 0.0001 --rate 0.01
./distribute.sh --server hydra1 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session SIGNAL_REAL --mu 45 --sigma 2 --a 30 --b 60 --lambda 0 --rate 0.01

