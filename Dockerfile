#####################################################
# Dockerfile
#####################################################

# Please uncomment the following command if you have a GPU
# The following base image contains: jax (with GPU support), numpy and scipy
# Nvidia documents will be located at /workspace
# FROM nvcr.io/nvdlfwea/jax/jax:23.05-py3

# If you are using a NVIDIA base image
# please remember to login into their server first
# Run the following command in your terminal
# docker login nvcr.io
# It will ask you for a username and a password. 
# The username is $oauthtoken
# The password is your API KEY (see: https://ngc.nvidia.com/setup/api-key )

# The following base image can be used if only CPU support is needed
FROM python:3.11-slim-buster
#  This image does not jax, numpy and scipy. Thus, we proceed to install them.
RUN pip install jax==0.4.23
RUN pip install jaxlib==0.4.23
RUN pip install scipy==1.11.4
RUN pip install numpy==1.26.2

# We install additional packages required for this particular project
# Do indicate which version you need so that the script is determistic. 
# If you do not specify the version, pip will try to install the lastest available
# version that is compatible with your other packages
RUN pip install jaxopt==0.8.1
RUN pip install pandas==2.1.1
RUN pip install tqdm==4.65.0
RUN pip install matplotlib==3.8.0
RUN pip install seaborn==0.13.0
RUN pip install scipy==1.11.4
RUN pip install distinctipy==1.2.3
RUN pip install statsmodels==0.14.0
RUN pip install cloudpickle==3.0.0

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






