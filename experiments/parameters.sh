#!/bin/bash

# hydra 7 and 8 are 126G
./distribute.sh --server hydra5 --method bin_mle --k 2 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra4 --method bin_mle --k 3 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra12 --method bin_mle --k 4 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra11 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra10 --method bin_mle --k 10 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra9 --method bin_mle --k 15 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra8 --method bin_mle --k 20 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra7 --method bin_mle --k 25 --std_signal_region 3 --data_id 50 --session PARAMETERS
./distribute.sh --server hydra6 --method bin_mle --k 30 --std_signal_region 3 --data_id 50 --session PARAMETERS

./distribute.sh --server hydra10 --method bin_mle --k 2 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra9 --method bin_mle --k 3 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra8 --method bin_mle --k 4 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra7 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra6 --method bin_mle --k 10 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra5 --method bin_mle --k 15 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra4 --method bin_mle --k 20 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra3 --method bin_mle --k 25 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra2 --method bin_mle --k 30 --std_signal_region 3 --data_id real --session PARAMETERS_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01