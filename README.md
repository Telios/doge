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

First you need to calibrate the robot.

```bash
bash scripts/calibrate.sh
```

To run doge with Solo12, use the following command:
```bash
bash scripts/run_doge.sh
```

Follow the instructions in the terminal to start doge.