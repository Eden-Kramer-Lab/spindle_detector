# Spindle Detector

Detect spindles using an HMM

### Installation ###
`spindle_detector` can be installed through pypi or conda. Conda is the best way to ensure that everything is installed properly.

```bash
pip install spindle_detector
python setup.py install
```
Or

```bash
conda install -c edeno spindle_detector
python setup.py install
```

### Usage ###
```python
results_df, model = detect_spindle(time, lfps, sampling_frequency)
```

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
conda activate spindle_detector # if using older anaconda, use source activate
python setup.py develop
```
