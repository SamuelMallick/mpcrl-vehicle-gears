#!/bin/bash

# Simple bash script to run a batch of evaluations for a single vehicle.

# Check if the right number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Missing argument. Usage: . run_eval_single.sh <script name> <seed_max> <config> <config_seed> <config_step>."
  echo "Either one, two, or five arguments can be provided."
  echo "1st argument: single vehicle script name (required)."
  echo "2nd argument: maximum seed value (optional, default is 10)."
  echo "3rd argument: configuration number (optional, default is 0 to make python choose)."
  echo "4th argument: configuration seed (optional, default is 0 to make python choose)."
  echo "5th argument: configuration step (optional, default is 0 to make python choose).".
  echo "Example: . run_eval_single.sh l_mpc.py 10 2 4 4000000"
  echo ""
  return
fi

# Get the first argument and check that the file exists
FILENAME=$1
if ! [ -f "run_single_vehicle/$FILENAME" ]; then
  echo "Error: File 'run_single_vehicle/$FILENAME' does not exist."
  return
fi

# Set the maximum seed value (if provided)
if [ "$#" -ge 2 ]; then
  SEED_MAX=$2
  if ! [[ "$SEED_MAX" =~ ^[0-9]+$ ]]; then
    echo "Error: Second argument must be an integer number (end seed)."
    return
  fi
else
  echo "SEED_MAX not provided. Using default value."
  SEED_MAX=10  # Default value if not provided
fi

# Set the configuration parameters
if [ "$#" -ge 3 ]; then
  if ! [ "$#" -eq 5 ]; then
    echo "Error: Expected 5 arguments, but got $#."
    return
  fi

  # If the right number of arguments is provided, assign them
  CONFIG=$3
  if ! [[ "$CONFIG" =~ ^[0-9]+$ ]]; then
    echo "Error: Third argument must be an integer number (CONFIG)."
    return
  fi
  CONFIG_SEED=$4
  if ! [[ "$CONFIG_SEED" =~ ^[0-9]+$ ]]; then
    echo "Error: Fourth argument must be an integer number (CONFIG_SEED)."
    return
  fi
  CONFIG_STEP=$5
  if ! [[ "$CONFIG_STEP" =~ ^[0-9]+$ ]]; then
    echo "Error: Fifth argument must be an integer number (CONFIG_STEP)."
    return
  fi
else
  echo "Config parameters not provided. Using default value from python script."
  CONFIG=0  # Null value to make python choose
  CONFIG_SEED=0
  CONFIG_STEP=0
fi

# Run evaluations
echo "Running evaluation of: run_single_vehicle/$FILENAME"
for SEED in $( eval echo {1..$SEED_MAX} ); do
  echo -e "\n########### SEED $SEED ###########"
  taskset -c 0-7 python "run_single_vehicle/$FILENAME" eval_seeds.eval_seed$SEED $CONFIG $CONFIG_SEED $CONFIG_STEP
done

echo -e "\nEvaluation completed."
