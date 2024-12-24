# Robust semi-parametric signal detection in particle physics with classifiers decorrelated via optimal transport

By Purvasha Chakravarti*, Lucas Kania*, Olaf Behnke, Mikael Kuusela, and
Larry Wasserman. (*) denotes equal contribution.

The provided code reproduces the experiments described
in https://arxiv.org/pdf/2409.06399.

## Abstract

Searches for new signals in particle physics are usually done by training a
supervised classifier to separate a signal model from the known Standard Model
physics (also called the background model). However, even when the signal model
is correct, systematic errors in the background model can influence supervised
classifiers and might adversely affect the signal detection procedure. To tackle
this problem, one approach is to use the (possibly misspecified) classifier only
to perform a preliminary signal-enrichment step and then to carry out a bump
hunt on the signal-rich sample. For this procedure to work, we need a classifier
constrained to be decorrelated with one or more protected variables used for the
signal-detection step. We do this by considering an optimal transport map of the
classifier output that makes it independent of the protected variable(s) for the
background. We then fit a semi-parametric mixture model to the distribution of
the protected variable after making cuts on the transformed classifier to detect
the presence of a signal. We compare and contrast this decorrelation method with
previous approaches, show that the decorrelation procedure is robust to moderate
background misspecification, and analyze the power and validity of the signal
detection test as a function of the cut on the classifier both with and without
decorrelation. We conclude that decorrelation and signal enrichment help produce
a stable, robust, valid, and more powerful test.

## Files

We note the following folders and files contained in this his repository.

| Name                                             | Description                                                                                                                    |
|--------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `cdot`                                           | Contains the necessary code to produce classifiers decorrelated via optimal transport (CDOT)                                   |
| [`rundocker.sh`](rundocker.sh)                   | Prepares docker environment to reproduce the power experiments with correlated and decorrelated classifiers.                   |
| `experiments`                                    | Source code to run power experiments.                                                                                          |
| [`experiments/power.py`](./experiments/power.py) | Main script called by rundocker.sh. It loads all the necesary data, creates the the test procedures and evaluates their power. |
| [`Dockerfile`](Dockerfile)                       | Configures the necessary dependencies to reproduce the power experiments.                                                      |
| `data`                                           | Data required to reproduce the power experiments.                                                                              |
| `src`                                            | Source code to create classifier-based test procedures.                                                                        |
| `results`                                        | The results of the power experiments are saved in this folder.                                                                 |

## Contents

- [Decorrelating classifiers via optimal transport](#decorrelating-classifiers-via-optimal-transport)
    - [Detection of high-p_T W-bosons experiments (WTagging)](#detection-of-high-p_t-w-bosons-experiments-wtagging)
        - [Comparing CDOT to existing decorrelation methods](#comparing-cdot-to-existing-decorrelation-methods)
    - [Detection of exotic high-mass resonance experiment (3b - 4b)](#detection-of-exotic-high-mass-resonance-experiment-3b---4b)
    - [Convert Rdata output to txt files](#convert-rdata-output-to-txt-files)
- [Evaluating the performance of test procedures based on correlated and decorrelated classifiers](#evaluating-the-performance-of-test-procedures-based-on-correlated-and-decorrelated-classifiers)
    - [Detection of high-p_T W-bosons experiments (WTagging)](#detection-of-high-p_t-w-bosons-experiments-wtagging-1)
    - [Detection of exotic high-mass resonance experiment (3b)](#detection-of-exotic-high-mass-resonance-experiment-3b)
    - [Detection of exotic high-mass resonance experiment (4b)](#detection-of-exotic-high-mass-resonance-experiment-4b)

## Decorrelating classifiers via optimal transport

You need R to run the commands in this section.
See https://www.r-project.org/ for instructions on how to install R.

### Detection of high-p_T W-bosons experiments (WTagging)

For the WTgging experiement, you can ontain the output of the correlated and
decorrelated classifiers by running the following commands.

```
unzip ./cdot/WTaggingTest.csv.zip -d ./cdot/
Rscript ./cdot/WTaggingCDOT.R
```

After the script finishes, the file `./cdot/WTaggingDecorrelated.Rdata` is
available. It contains the training, validation and test data sets.
Each data set has a column
`h` that provides the classifier output and a column `Trans_h`
that provides the decorrelated classifier output for each observation.
Alternatevely, you can obtain the file
`./cdot/WTaggingDecorrelated.Rdata` by running the following command.

```
unzip ./cdot/WTaggingDecorrelated.Rdata.zip -d ./cdot/
```

To produce figures 1, 3 and 4, run the following command.

```
Rscript ./cdot/WTaggingPlots.R
```

After the script finishes, the figures are available at the following locations.

| Figure number (click link to open)            | Location                                   |
|-----------------------------------------------|--------------------------------------------|
| [1a](./cdot/img/PlotMass.png)                 | `./cdot/img/PlotMass.png`                  |
| [1b](./cdot/img/PlotMassCut.png)              | `./cdot/img/PlotMassCut.png`               |
| [3](./cdot/img/WTaggingNoDecorrelation.png)   | `./cdot/img/WTaggingNoDecorrelation.png`   |
| [4](./cdot/img/WTaggingWithDecorrelation.png) | `./cdot/img/WTaggingWithDecorrelation.png` |

#### Comparing CDOT to existing decorrelation methods

The code in `./cdot/WTaggingCDOT.R` also outputs an additional file
`./cdot/DataforR50JSDPlot.Rdata`, which can be used to calculate the 1/JSD and
the R50 scores needed for Figure 5. To ensure that the same bins are used as in
Figure 7 of Kitouni et al. (2021), run the Python code in

```
jupyter notebook ./cdot/R50JSDPlot.ipynb
```

The script stores the scores in `PythonR50JSDPlotData.RData`. Finally, to draw
figure 5, run the following command.

```
Rscript ./cdot/R50JSDPlot.R
```

After the script finishes, the figure is available at the following location.

| Figure number (click link to open)  | Location                         |
|-------------------------------------|----------------------------------|
| [5](./cdot/img/R50JSDPlotFinal.png) | `./cdot/img/R50JSDPlotFinal.png` |

### Detection of exotic high-mass resonance experiment (3b - 4b)

For 3b and 4b experiments, the following script performs some variable
transformations to improve the overall performance of the classifiers.

```
Rscript ./cdot/DataAnalysis3b4b.R
```

After the script finishes, the file `./cdot/Data3b4b.RData` is available. It
provides the 3b, 4b and signal data sets used in the experiments. We proceed to
randomly select the
training, validation and test data sets from the entire 3b, 4b and signal data
sets to produce the outputs of the correlated and decorrelated classifiers.

```
Rscript ./cdot/3b4bCDOT.R
```

After the script finishes, the file `./cdot/DecorrelatedData3b4b.Rdata` is
available. Its contains the training,
validation and test data sets, together with
a column `h` that provides the classifier output and a column
`Trans_h` that provides the output of the decorrelated classifier.

To produce figures 9 and 10, run the following command.

```
Rscript ./cdot/3b4bPlots.R
```

After the script finishes, they are available at the following locations.

| Figure number (click link to open)         | Location                               |
|--------------------------------------------|----------------------------------------|
| [9](./cdot/img/4bNoDecorrelation.png)      | `./cdot/img/4bNoDecorrelation.png`     |
| [10](./cdot/img/3b4bWithDecorrelation.png) | `./cdot/img/3b4bWithDecorrelation.png` |

### Convert Rdata to txt files

If the file `./cdot/WTaggingDecorrelated.Rdata` is not in the folder `./cdot/`,
run the following command to obtain it.

```
unzip ./cdot/WTaggingTest.csv.zip -d ./cdot/
```

To convert the `.Rdata` to `.txt` files, run the following command.

```
Rscript ./cdot/convert.R
```

## Evaluating the performance of test procedures based on correlated and decorrelated classifiers

You need Docker to run the commands in this section.
See https://docs.docker.com/desktop/ for
instructions on how to install Docker with a graphical interface.

### Detection of high-P_T W-bosons experiments (WTagging)

Run the following command.

```
./rundocker.sh --data_id WTagging --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                        | Description                                                                                       | Location                                               |
|-----------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------------|
| [14](./results/WTagging/val/class/0.0/selection.pdf)      | Selection of test statistic that achieves best type I error                                       | `./results/WTagging/val/class/0.0/selection.pdf`       |
| [15](./results/WTagging/val/class/0.0/pvalues.pdf)        | Distribution of p-values corresponding the above test statistics                                  | `./results/WTagging/val/class/0.0/pvalues.pdf`         |
| [6](./results/WTagging/test/35/power.pdf)                 | Power comparison of classifier-based test procedures using decorrelated and correlated classfiers | `./results/WTagging/test/35/power.pdf`                 |
| [7](./results/WTagging/test/35/class_filter_uniform.pdf)  | Background fit used by test procedure with correlated classifier                                  | `./results/WTagging/test/35/class_filter_uniform.pdf`  |
| [8](./results/WTagging/test/35/tclass_filter_uniform.pdf) | Background fit used by test procedure with decorrelated classifier                                | `./results/WTagging/test/35/tclass_filter_uniform.pdf` |

### Detection of exotic high-mass resonance experiment (3b)

Run the following command.

```
./rundocker.sh --data_id 3b --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                   | Description                                                                                       | Location                                         |
|------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [16](./results/3b/val/class/0.0/selection.pdf)       | Selection of test statistic that achieves best type I error                                       | `./results/3b/val/class/0.0/selection.pdf`       |
| [17](./results/3b/val/class/0.0/pvalues.pdf)         | Distribution of p-values corresponding the above test statistics                                  | `./results/3b/val/class/0.0/pvalues.pdf`         |
| [11](./results/3b/test/20/power.pdf)                 | Power comparison of classifier-based test procedures using decorrelated and correlated classfiers | `./results/3b/test/20/power.pdf`                 |
| [18](./results/3b/test/20/class_filter_uniform.pdf)  | Background fit used by test procedure with correlated classifier                                  | `./results/3b/test/20/class_filter_uniform.pdf`  |
| [19](./results/3b/test/20/tclass_filter_uniform.pdf) | Background fit used by test procedure with decorrelated classifier                                | `./results/3b/test/20/tclass_filter_uniform.pdf` |

### Detection of exotic high-mass resonance experiment (4b)

Run the following command.

```
./rundocker.sh --data_id 4b --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                   | Description                                                                                       | Location                                         |
|------------------------------------------------------|---------------------------------------------------------------------------------------------------|--------------------------------------------------|
| [12](./results/4b/test/20/power.pdf)                 | Power comparison of classifier-based test procedures using decorrelated and correlated classfiers | `./results/4b/test/20/power.pdf`                 |
| [20](./results/4b/test/20/class_filter_uniform.pdf)  | Background fit used by test procedure with correlated classifier                                  | `./results/4b/test/20/class_filter_uniform.pdf`  |
| [21](./results/4b/test/20/tclass_filter_uniform.pdf) | Background fit used by test procedure with decorrelated classifier                                | `./results/4b/test/20/tclass_filter_uniform.pdf` |
