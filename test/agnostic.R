options(pkgType = "binary")
install.packages("reticulate", repos = 'http://cran.us.r-project.org')
library(reticulate)

tryCatch({
  install_miniconda(path = miniconda_path(), update = FALSE, force = FALSE) },
  error = function(e) {
    print("Miniconda installed")
  })

envname <- "delta_test"

packages <- c('jax==0.4.16',
              'jaxopt==0.8.1',
              'numpy==1.26.0',
              'scipy==1.11.3',
              'tqdm==4.66.1')

# conda_remove(envname)
conda_create(envname = envname,
             packages = packages,
             python_version = "3.11.6",
             forge = TRUE)

use_condaenv(envname)

source('')




