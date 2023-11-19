#!/bin/bash

# Generate a random seed
SEED=$RANDOM
echo $SEED
# Submit the job with the job name and output file name set to include the seed
sbatch --job-name="bayesian_$SEED" --output="bayesian_$SEED.out" bayesian.SBATCH $SEED
