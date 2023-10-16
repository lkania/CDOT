#####################################################
# Dockerfile
#####################################################
# If you are using an NVIDIA base image 
# please remember to login into their server first
# Run the following command in your terminal
# docker login nvcr.io
# It will ask you for a username and a password. 
# The username is $oauthtoken
# The password is your API KEY (see: https://ngc.nvidia.com/setup/api-key )

# The base image contains: jax, numpy and scipy
# Nvidia documents will be located at /workspace
FROM nvcr.io/nvdlfwea/jax/jax:23.05-py3

# We install additional packages required for this particular project
# Do indicate which version you need so that the script is determistic. 
# If you do not specify the version, pip will try to install the lastest available
# version that is compatible with your other packages
RUN pip install jaxopt==0.7
RUN pip install pandas==2.0.3
RUN pip install tqdm==4.65.0

# We copy the folder containing the source code of our algorithm 
# into the folder /program/src in the container
COPY src /program/src

# We copy the folder containing the scripts to run experiments with our algorithm
# into the folder /program/experiments in the container
COPY experiments /program/experiments

# If you want to exclude any files inside the above directories
# You should add the exeptions to .dockerignore

# We set the working directory to program
# So that bash starts at /program when, we loggin into the container
WORKDIR /program






