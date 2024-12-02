options(repos = c(CRAN = "https://cloud.r-project.org/"))

getPackage <- function(package) {
  if (!requireNamespace(package, quietly = TRUE)) {
    message(paste("Installing package:", package))
    install.packages(package)
  }
  message(paste("Package", package, "is ready to use."))
}

getBiocPackage <- function(package) {
  getPackage('BiocManager')
  if (!requireNamespace(package, quietly = TRUE)) {
    message(paste("Installing package:", package))
    BiocManager::install(package)
  }
  message(paste("Package", package, "is ready to use."))
}

# computation
getPackage('np')
getPackage('ranger')
getBiocPackage('rhdf5')

# plots
getPackage('ggplot2')
getPackage('ggpubr')
getPackage('gridExtra')
getPackage('knitr')
getPackage('colorBlindness')
getPackage('latex2exp')
getPackage('dplyr')
getPackage('tidyr')


