#!/bin/bash

# Check if an argument was provided
if [ "$#" -lt 1 ]; then
  echo "Missing argument. Usage: . run_eval_single.sh <argument>"
  return
fi
if [ "$#" -gt 1 ]; then
  echo "Too many arguments. Usage: . run_eval_single.sh <argument>"
  return
fi
if [ ! -f "run_single_vehicle/$ARGUMENT" ]; then
    echo "Error: File 'run_single_vehicle/$ARGUMENT' does not exist."
    return
fi

# Get the first argument
ARGUMENT="$1"

# Run evaluations
echo "Running evaluation of: run_single_vehicle/$ARGUMENT"
echo "\r########### SEED 1 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed1
echo "\r########### SEED 2 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed2
echo "\r########### SEED 3 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed3
echo "\r########### SEED 4 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed4
echo "\r########### SEED 5 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed5