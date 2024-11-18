# Robust semi-parametric signal detection in particle physics with classifiers decorrelated via optimal transport

By Purvasha Chakravarti*, Lucas Kania*, Olaf Behnke, Mikael Kuusela, and
Larry Wasserman. (*) Denotes equal contribution.

The provided code reproduces the experiments described
in https://arxiv.org/pdf/2409.06399.

Files for reproducing the decorrelation experiments in "Robust semi-parametric
signal detection in particle physics with classifiers decorrelated via optimal
transport".

## Install required packages

Execute the following command to install the necessary packages

```{r, eval=FALSE}
source("requirements.R")
```

# Experiments

The code for the decorrelation experiments in the paper can be found here.

First, load the file with functions required to train the CDOT algorithm and
predict it on new data.

```{r, eval=FALSE}
source("CDOTFunctions.R")
```

The rest of the code is specific to the two different experiments in the paper.

## Detection of high-$p_{\mathrm{T}}$ W bosons experiments (WTagging)

To get the classifier as well as the decorrelated classifier outputs for the
WTagging experiment run the following file:

```{r, eval=FALSE}
source("WTaggingCDOT.R")
```

The code outputs a file called `WTaggingDecorrelated.Rdata` that contains
the training, validation and test data sets. Each data set has a column
`h` that provides the classifier output and a column `Trans_h`
that provides the transformed decorrelated classifier output for the
corresponding data set.

Using these data sets, Figures 1, 3 and 4 can be obtained by running the
following:

```{r, eval=FALSE}
source("WTaggingFinalPlots.R")
```

### Comparing CDOT to existing decorrelation methods: Figure 5

The code in `WTaggingCDOT.R` also outputs an additional file
`DataforR50JSDPlot.Rdata`, which can be used to calculate the 1/JSD and
the R50 scores for Figure 5. To ensure that the same bins are used for the
comparison as used in Figure 7 of Kitouni et al. (2021), we use a similar python
code as Kitouni et al. (2021) in `R50JSDPlot.ipynb` which outputs the
scores in `PythonR50JSDPlotData.RData`. We finally plot Figure 5 using
the following:

```{r, eval=FALSE}
source("R50JSDPlot.R")
```

## Detection of exotic high-mass resonance (3b - 4b experiment)

For this experiment, we first need to perform some variable transformations to
improve the overall performance of the classifiers. We do this as follows:

```{r, eval=FALSE}
source("DataAnalysis3b4b.R")
```

The code above outputs a file called `Data3b4b.RData` that provides the
3b, 4b and signal data sets used in the experiments. We now randomly select the
training, validation and test data sets from the entire 3b, 4b and signal data
sets and get the classifier as well as the decorrelated classifier outputs for
the experiment using the following:

```{r, eval=FALSE}
source("DataAnalysis3b4b.R")
```

This outputs `DecorrelatedData3b4b.Rdata` that contains the training,
validation and test data sets for both backgrounds and the signal. Each data set
has a column `h` that provides the classifier output and a column
`Trans_h` that provides the transformed decorrelated classifier output
for the corresponding data set. Using these data sets, we plot Figures 9 and 10
using

```{r, eval=FALSE}
source("FinalPlots3b4b.R")
```
