#!/bin/bash

# Usage: ./run.sh [dataset_name]
# Example: ./run.sh starbucks
# Default: hillstrom

DATASET=${1:-hillstrom}

export PYTHONPATH=src
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.01
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.02
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.03
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.04
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.05
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.06
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.07
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.08
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.09
python src/experiment/main.py data=$DATASET model=lgbm experiment.frac=0.1

python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.01
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.02
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.03
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.04
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.05
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.06
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.07
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.08
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.09
python src/experiment/main.py data=$DATASET model=mlp experiment.frac=0.1

python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.01
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.02
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.03
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.04
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.05
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.06
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.07
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.08
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.09
python src/experiment/main.py data=$DATASET model=uplift_rf experiment.frac=0.1
