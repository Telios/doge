#!/bin/bash

pip install external/rpg_vid2e/esim_py
pip install external/rpg_vid2e/esim_torch

cd external/IEBCS/cpp && ./compile_test.sh