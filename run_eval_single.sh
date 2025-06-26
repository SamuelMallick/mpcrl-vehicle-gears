#!/bin/bash

# Check if the right number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Missing argument. Usage: . run_eval_single.sh <argument>"
  return
fi
if [ "$#" -gt 1 ]; then
  echo "Too many arguments. Usage: . run_eval_single.sh <argument>"
  return
fi

# Get the first argument and check that the file exists
ARGUMENT=$1
if ! [ -f "run_single_vehicle/$ARGUMENT" ]; then
  echo "Error: File 'run_single_vehicle/$ARGUMENT' does not exist."
  return
fi

# Run evaluations
echo "Running evaluation of: run_single_vehicle/$ARGUMENT"
echo -e "\n########### SEED 1 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed1
echo -e "\n########### SEED 2 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed2
echo -e "\n########### SEED 3 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed3
echo -e "\n########### SEED 4 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed4
echo -e "\n########### SEED 5 ###########"
python "run_single_vehicle/$ARGUMENT" eval_seeds.eval_seed5

echo -e "\nEvaluation completed."

# Clean up the argument variable
unset ARGUMENT
