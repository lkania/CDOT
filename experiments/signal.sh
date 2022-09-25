#!/bin/bash

# load functions
. distribute.sh
# TODO: currently failling
# hydra 7 and 8 are 126G
send hydra7 bin_mle None 30 3 True 50

#send hydra5 bin_mle None 30 2 True
#send hydra4 bin_mle None 30 1 True
#
#send hydra3 bin_mom lawson_scipy 30 3 True
#send hydra2 bin_mom lawson_scipy 30 2 True
#send hydra1 bin_mom lawson_scipy 30 1 True