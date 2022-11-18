#!/bin/bash

./distribute.sh --server hydra12 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session CONTAMINATION
./distribute.sh --server hydra11 --method bin_mle --k 5 --std_signal_region 2.5 --data_id 50 --session CONTAMINATION
./distribute.sh --server hydra10 --method bin_mle --k 5 --std_signal_region 2 --data_id 50 --session CONTAMINATION
./distribute.sh --server hydra9 --method bin_mle --k 5 --std_signal_region 1.5 --data_id 50 --session CONTAMINATION
./distribute.sh --server hydra8 --method bin_mle --k 5 --std_signal_region 1 --data_id 50 --session CONTAMINATION
./distribute.sh --server hydra7 --method bin_mle --k 5 --std_signal_region 0.5 --data_id 50 --session CONTAMINATION

./distribute.sh --server hydra12 --method bin_mle --k 5 --std_signal_region 3 --data_id real --session CONTAMINATION_REAL --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra11 --method bin_mle --k 5 --std_signal_region 2.5 --data_id real --session CONTAMINATION_REAL  --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra10 --method bin_mle --k 5 --std_signal_region 2 --data_id real --session CONTAMINATION_REAL  --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra9 --method bin_mle --k 5 --std_signal_region 1.5 --data_id real --session CONTAMINATION_REAL  --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra8 --method bin_mle --k 5 --std_signal_region 1 --data_id real --session CONTAMINATION_REAL  --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
./distribute.sh --server hydra7 --method bin_mle --k 5 --std_signal_region 0.5 --data_id real --session CONTAMINATION_REAL  --mu 45 --sigma 2 --a 30 --b 60 --rate 0.01
