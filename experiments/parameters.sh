#!/bin/bash

# load functions
. distribute.sh

# hydra 7 and 8 are 126G
send hydra12 bin_mle None 2 3 False 50
send hydra11 bin_mle None 3 3 False 50
send hydra10 bin_mle None 4 3 False 50
#send hydra12 bin_mle None 5 3 False 50
#send hydra11 bin_mle None 10 3 False 50
#send hydra10 bin_mle None 20 3 False 50
#send hydra9 bin_mle None 30 3 False 50
#send hydra4 bin_mle None 40 3 False 50

#send hydra8 bin_mom None 5 3 False 50
#send hydra7 bin_mom None 10 3 False 50
#send hydra6 bin_mom None 20 3 False 50
#send hydra5 bin_mom None 30 3 False 50
#send hydra3 bin_mom None 40 3 False 50