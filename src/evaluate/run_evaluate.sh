#!/bin/bash
# Wrapper: install xgboost then run evaluate.py
# SageMaker ScriptProcessor with command=["bash"] runs this directly.
set -e
pip install xgboost>=1.7.0 -q
python /opt/ml/processing/input/code/evaluate.py "$@"
