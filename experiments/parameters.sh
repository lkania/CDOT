#!/bin/bash

# load functions
. distribute.sh

# hydra 7 and 8 are 126G
send hydra11 bin_mle None 5 3 False 50
#send hydra11 bin_mle None 10 3 False
send hydra9 bin_mle None 20 3 False 50
send hydra8 bin_mle None 30 3 False 50
send hydra7 bin_mle None 50 3 False 50
#send hydra7 bin_mle None 100 3 False


#send hydra8 bin_mom lawson_scipy 5 3 False
#send hydra7 bin_mom lawson_scipy 10 3 False
#send hydra6 bin_mom lawson_scipy 20 3 False
#send hydra7 bin_mom lawson_scipy 30 3 False 50