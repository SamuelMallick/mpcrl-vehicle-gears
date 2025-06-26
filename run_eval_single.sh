#!/bin/bash

# Check if the right number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Missing argument. Usage: . run_eval_single.sh <filename> <end_seed (optional)>"
  return
fi
if [ "$#" -gt 2 ]; then
  echo "Too many arguments. Usage: . run_eval_single.sh <filename> <end_seed (optional)>"
  return
fi

# Get the first argument and check that the file exists
FILENAME=$1
if ! [ -f "run_single_vehicle/$FILENAME" ]; then
  echo "Error: File 'run_single_vehicle/$FILENAME' does not exist."
  return
fi

# Set the maximum seed value
if [ "$#" -eq 2 ]; then
  SEED_MAX=$2
  if ! [[ "$SEED_MAX" =~ ^[0-9]+$ ]]; then
    echo "Error: Second argument must be an integer number (end seed)."
    return
  fi
else
  echo "SEED_MAX not provided. Using default value."
  SEED_MAX=10  # Default value if not provided
fi

# Run evaluations
echo "Running evaluation of: run_single_vehicle/$FILENAME"
for SEED in $( eval echo {1..$SEED_MAX} ); do
  echo -e "\n########### SEED $SEED ###########"
  taskset -c 0-7 python "run_single_vehicle/$FILENAME" eval_seeds.eval_seed$SEED
done

echo -e "\nEvaluation completed."

# Clean up the argument variable
unset FILENAME
unset SEED_MAX
unset SEED