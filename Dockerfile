#####################################################
# Dockerfile
#####################################################

# The following base image can be used if only CPU support is needed
FROM python:3.11-slim-buster
#  This image does not have jax, numpy, and scipy. Thus, we proceed to install them.
RUN pip install jax==0.4.23
RUN pip install jaxlib==0.4.23
RUN pip install scipy==1.11.4
RUN pip install numpy==1.26.2

# We install additional packages required for this particular project
# We specify which version is needed to make the script deterministic. 
RUN pip install jaxopt==0.8.1
RUN pip install pandas==2.1.1
RUN pip install tqdm==4.65.0
RUN pip install matplotlib==3.8.0
RUN pip install seaborn==0.13.0
RUN pip install scipy==1.11.4
RUN pip install statsmodels==0.14.0
RUN pip install cloudpickle==3.0.0

# We copy the folder containing the source code of our algorithm 
# into the folder /program/src in the container
COPY src /program/src

# We copy the folder containing the scripts to run experiments with our algorithm
# into the folder /program/experiments in the container
COPY experiments /program/experiments

# If you want to exclude any files inside the above directories
# You should add the exceptions to .dockerignore

# We set the working directory to the program
# So that bash starts at /program when we log in to the container
WORKDIR /program