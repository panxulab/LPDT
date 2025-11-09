#!/bin/bash

cd envs
pip install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
cd jacopinpad
pip install -e .
cd ..
cd metaworld
pip install -e .
cd ..
cd mujoco-control-envs
pip install -e .
cd ../..