#!/bin/bash

# hydra 7 and 8 are 126G
./distribute.sh --server hydra7 --method bin_mle --k 5 --std_signal_region 3 --data_id 100 --session SAMPLE
./distribute.sh --server hydra12 --method bin_mle --k 5 --std_signal_region 3 --data_id 50 --session SAMPLE
./distribute.sh --server hydra11 --method bin_mle --k 5 --std_signal_region 3 --data_id 5 --session SAMPLE
./distribute.sh --server hydra10 --method bin_mle --k 5 --std_signal_region 3 --data_id half --session SAMPLE

