# Robust semi-parametric signal detection in particle physics with classifiers decorrelated via optimal transport

By Purvasha Chakravarti*, Lucas Kania*, Olaf Behnke, Mikael Kuusela, and
Larry Wasserman. (*) Denotes equal contribution.

The provided code reproduces the experiments described
in https://arxiv.org/pdf/2409.06399.

We remark the following folder and files contained in this his repository.

| Name                                             | Description                                                                                                                                  |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| [`cdot`](./cdot/README.md)                       | Contains the necessary code to decorrelate classifiers via optimal transport (CDOT). [See `./cdot/README.md` for details](./cdot/README.md). |
| [`rundocker.sh`](rundocker.sh)                   | Prepares docker environment to reproduce the power experiments with correlated and decorrelated classifiers.                                 |
| `experiments`                                    | Source code to run power experiments.                                                                                                        |
| [`experiments/power.py`](./experiments/power.py) | Main script called by rundocker.sh. It loads all the necesary data, creates the the test procedures and evaluates their power.               |
| [`Dockerfile`](Dockerfile)                       | Configures the necessary dependencies to reproduce the power experiments.                                                                    |
| `data`                                           | Data required to reproduce the power experiments.                                                                                            |
| `src`                                            | Source code to create classifier-based test procedures.                                                                                      |
| `results`                                        | The results of the power experiments are saved in this folder.                                                                               |

## Requirements

Please install Docker in your system; see https://docs.docker.com/desktop/ for
instructions on how to install Docker with a graphical interface.

## Detection of high-$p_{\mathrm{T}}$ W-bosons experiments (WTagging)

Run the following commands.

```
chmod +x rundocker.sh
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

## Detection of exotic high-mass resonance experiment (3b)

Run the following commands.

```
chmod +x rundocker.sh
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

## Detection of exotic high-mass resonance experiment (4b)

Run the following commands.

```
chmod +x rundocker.sh
./rundocker.sh --data_id 4b --cwd .
```

After the script finishes, the following figures are available.

| Figure number (click link to open)                   | Location                                         |
|------------------------------------------------------|--------------------------------------------------|
| [12](./results/4b/test/20/power.pdf)                 | `./results/4b/test/20/power.pdf`                 |
| [20](./results/4b/test/20/class_filter_uniform.pdf)  | `./results/4b/test/20/class_filter_uniform.pdf`  |
| [21](./results/4b/test/20/tclass_filter_uniform.pdf) | `./results/4b/test/20/tclass_filter_uniform.pdf` |
