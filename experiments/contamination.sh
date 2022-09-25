#!/bin/bash

# load functions
. distribute.sh

# hydra 7 and 8 are 126G
send hydra12 bin_mle None 30 3  False
send hydra11 bin_mle None 30 2  False
send hydra10 bin_mle None 30 1  False

send hydra9 bin_mom lawson_scipy 30 3  False
send hydra8 bin_mom lawson_scipy 30 2  False
send hydra7 bin_mom lawson_scipy 30 1  False