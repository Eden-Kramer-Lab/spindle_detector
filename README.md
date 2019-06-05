# Spindle Detector

Detect spindles using an HMM

### Developer Installation ###
1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Go to the local repository (`.../spindle_detector`) and install the anaconda environment for the repository. Type into bash:
```bash
conda env create -f environment.yml
source activate spindle_detector
python setup.py develop
```
