# DOGE : Dynamic Obstacle Ground-based Evasion

## Installation
Clone the repository using the following command:
```bash
git clone --recurse-submodules https://github.com/Telios/doge.git
```

Make a new conda environment and install the required packages using the following commands:
```bash
conda env create -f environment.yml
conda activate doge
```

Install the external libraries
```bash
./install_external.sh
```

## Usage
To run the code, use the following command:
```bash
python testing/main.py
```
You should see a window pop up with the simulation running.