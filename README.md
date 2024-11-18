# Robust semi-parametric signal detection in particle physics with classifiers decorrelated via optimal transport

By Purvasha Chakravarti*, Lucas Kania*, Olaf Behnke, Mikael Kuusela, and
Larry Wasserman. (*) denotes equal contribution.

The provided code reproduces the experiments described
in https://arxiv.org/pdf/2409.06399.

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

## Training and decorrelating classifiers via optimal transport

You need R to run the commands in this section.
See https://www.r-project.org/ for instructions on how to install R.

### Detection of high-$p_{\mathrm{T}}$ W-bosons experiments (WTagging)

To get the classifier as well as the decorrelated classifier outputs for the
WTagging experiment run the following command.

```
Rscript ./cdot/WTaggingCDOT.R
```

The code outputs a file called `./cdot/WTaggingDecorrelated.Rdata` that contains
the training, validation and test data sets. Each data set has a column
`h` that provides the classifier output and a column `Trans_h`
that provides the transformed decorrelated classifier output for the
corresponding data set. The, we can proceed to run the following command.

```
Rscript ./cdot/WTaggingPlots.R
```

After the script finishes, the following figures are available.

| Figure number (click link to open)            | Location                                   |
|-----------------------------------------------|--------------------------------------------|
| [1a](./cdot/img/PlotMass.png)                 | `./cdot/img/PlotMass.png`                  |
| [1b](./cdot/img/PlotMassCut.png)              | `./cdot/img/PlotMassCut.png`               |
| [3](./cdot/img/WTaggingNoDecorrelation.pdf)   | `./cdot/img/WTaggingNoDecorrelation.pdf`   |
| [4](./cdot/img/WTaggingWithDecorrelation.png) | `./cdot/img/WTaggingWithDecorrelation.png` |

#### Comparing CDOT to existing decorrelation methods

The code in `WTaggingCDOT.R` also outputs an additional file
`DataforR50JSDPlot.Rdata`, which can be used to calculate the 1/JSD and
the R50 scores for Figure 5. To ensure that the same bins are used for the
comparison as used in Figure 7 of Kitouni et al. (2021), we use a similar python
code as Kitouni et al. (2021) in `R50JSDPlot.ipynb` which outputs the
scores in `PythonR50JSDPlotData.RData`. Finally, we run the following command.

```
Rscript ./cdot/R50JSDPlot.R
```

After the script finishes, the following figure is available.

| Figure number (click link to open)  | Location                         |
|-------------------------------------|----------------------------------|
| [5](./cdot/img/R50JSDPlotFinal.png) | `./cdot/img/R50JSDPlotFinal.png` |

### Detection of exotic high-mass resonance experiment (3b - 4b)

For this experiment, we first need to perform some variable transformations to
improve the overall performance of the classifiers. We do this as follows:

```
Rscript ./cdot/DataAnalysis3b4b.R
```

The code above outputs a file called `Data3b4b.RData` that provides the
3b, 4b and signal data sets used in the experiments. We now randomly select the
training, validation and test data sets from the entire 3b, 4b and signal data
sets and get the classifier as well as the decorrelated classifier outputs for
the experiment using the following command.

```
Rscript ./cdot/3b4bPlots.R
```

This outputs `DecorrelatedData3b4b.Rdata` that contains the training,
validation and test data sets for both backgrounds and the signal. Each data set
has a column `h` that provides the classifier output and a column
`Trans_h` that provides the transformed decorrelated classifier output
for the corresponding data set. Finally, run the following command.

```
Rscript ./cdot/3b4bPlots.R
```

After the script finishes, the following figures are available.

| Figure number (click link to open)         | Location                               |
|--------------------------------------------|----------------------------------------|
| [9](./cdot/img/4bNoDecorrelation.png)      | `./cdot/img/4bNoDecorrelation.png`     |
| [10](./cdot/img/3b4bWithDecorrelation.png) | `./cdot/img/3b4bWithDecorrelation.png` |

### Convert Rdata output to txt files

```
Rscript ./cdot/convert.R
```

## Evaluating the performance of decorrelated and correlated classifier-based test procedures

You need Docker to run the commands in this section.
See https://docs.docker.com/desktop/ for
instructions on how to install Docker with a graphical interface.

### Detection of high-$p_{\mathrm{T}}$ W-bosons experiments (WTagging)

Run the following command.

```
./rundocker.sh --data_id WTagging --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                        | Location                                               |
|-----------------------------------------------------------|--------------------------------------------------------|
| [6](./results/WTagging/test/35/power.pdf)                 | `./results/WTagging/test/35/power.pdf`                 |
| [7](./results/WTagging/test/35/class_filter_uniform.pdf)  | `./results/WTagging/test/35/class_filter_uniform.pdf`  |
| [8](./results/WTagging/test/35/tclass_filter_uniform.pdf) | `./results/WTagging/test/35/tclass_filter_uniform.pdf` |
| [14](./results/WTagging/val/class/0.0/selection.pdf)      | `./results/WTagging/val/class/0.0/selection.pdf`       |
| [15](./results/WTagging/val/class/0.0/pvalues.pdf)        | `./results/WTagging/val/class/0.0/pvalues.pdf`         |

### Detection of exotic high-mass resonance experiment (3b)

Run the following command.

```
./rundocker.sh --data_id 3b --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                   | Location                                         |
|------------------------------------------------------|--------------------------------------------------|
| [11](./results/3b/test/20/power.pdf)                 | `./results/3b/test/20/power.pdf`                 |
| [16](./results/3b/val/class/0.0/selection.pdf)       | `./results/3b/val/class/0.0/selection.pdf`       |
| [17](./results/3b/val/class/0.0/pvalues.pdf)         | `./results/3b/val/class/0.0/pvalues.pdf`         |
| [18](./results/3b/test/20/class_filter_uniform.pdf)  | `./results/3b/test/20/class_filter_uniform.pdf`  |
| [19](./results/3b/test/20/tclass_filter_uniform.pdf) | `./results/3b/test/20/tclass_filter_uniform.pdf` |

### Detection of exotic high-mass resonance experiment (4b)

Run the following command.

```
./rundocker.sh --data_id 4b --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                   | Location                                         |
|------------------------------------------------------|--------------------------------------------------|
| [12](./results/4b/test/20/power.pdf)                 | `./results/4b/test/20/power.pdf`                 |
| [20](./results/4b/test/20/class_filter_uniform.pdf)  | `./results/4b/test/20/class_filter_uniform.pdf`  |
| [21](./results/4b/test/20/tclass_filter_uniform.pdf) | `./results/4b/test/20/tclass_filter_uniform.pdf` |
