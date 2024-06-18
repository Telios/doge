# DOGE : Dynamic Obstacle Ground-based Evasion

## Installation
Clone the repository using the following command:
```bash
git clone --recurse-submodules https://github.com/Telios/doge.git
```

Make a new conda environment and install the required packages using the following commands:
```bash
conda create -n doge python=3.11
conda activate doge
pip install -r requirements.txt
pip install -r external/rpg_vid2e/requirements.txt
conda install -y -c conda-forge pybind11 matplotlib
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install the external libraries
```bash
pip install external/rpg_vid2e/esim_py
pip install external/rpg_vid2e/esim_torch
```

## Usage
To run the code, use the following command:
```bash
python testing/main.py
```
You should see a window pop up with the simulation running.