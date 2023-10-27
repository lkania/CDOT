###############################################
# Name of the desired python environment
###############################################
envname <- "delta"

is_available <- require("reticulate")
if (!is_available) {
  ###############################################
  # Installing a preparing python environment
  ###############################################
  options(pkgType = "binary")
  install.packages(
    "reticulate",
    repos = 'http://cran.us.r-project.org')
  library(reticulate)

  tryCatch({
    install_miniconda(path = miniconda_path(),
                      update = FALSE,
                      force = FALSE) },
    error = function(e) {
      print("Miniconda installed")
    })

  packages <- c('jax==0.4.16',
                'jaxopt==0.8.1',
                'numpy==1.26.0',
                'scipy==1.11.3',
                'tqdm==4.66.1')

  conda_create(envname = envname,
               packages = packages,
               python_version = "3.11.6",
               forge = TRUE)
}
###############################################
# Using python environment
###############################################
use_condaenv(envname)

source_python('./src/test/test.py')
source_python('./src/load.py')

###############################################
# test configuration
###############################################
args <- DotDic()
args$seed <- 0

###############################################
# background optimization
###############################################
args$method <- 'bin_mle'
args$bins <- 100
args$optimizer <- 'dagostini'
args$fixpoint <- 'normal'
args$maxiter <- 100
args$tol <- 1e-6

###############################################
# signal modelling
###############################################
args$model_signal <- FALSE

###############################################
# data transformation
###############################################
args$rate <- 0.003
args$a <- 201
args$b <- NULL
args$k <- 3
args$ks <- NULL

###############################################
# signal region
###############################################
args$lower <- 344
args$upper <- 446

###############################################
# load data
###############################################
# the following file contains 3000 observations
data <- load("./data/3b/demo.txt")

###############################################
# run test
###############################################
method <- test(args = args, X = data)

###############################################
# get p-value
###############################################
method$pvalue


