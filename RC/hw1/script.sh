#!/bin/bash

VENV_PATH="../../venv"

source "$VENV_PATH/bin/activate"

cd SuperGluePretrainedNetwork

./match_pairs.py --resize -1 --superglue indoor --max_keypoints 1024 --nms_radius 4 --input_dir ../superglue_input --output_dir ../superglue_result --input_pairs ../superglue_input/pairs

deactivate