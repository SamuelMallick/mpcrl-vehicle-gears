#!/bin/bash

DEBUG=false  # Set to true to enable debug mode

# Help message
print_help() {
  echo "Bash script to train a RL policy for a single vehicle."
  echo "Usage: $0 <config-file> [options]"
  echo
  echo "Positional arguments:"
  echo "  config_file <str>       Configuration file to use"
  echo
  echo "Options:"
  echo "  -p, --policy <str>      Folder where the starting policy to refine is stored (takes effect only if if configuration file is c2 or c4)"
  echo "  -s, --seed <int>        Seed value for the training (overrides the seed in the configuration file)"
  echo "  -h, --help              Show this help message"
  echo
  echo "Example:"
  echo " $0 c4_seed1.py --policy c3_seed5"
}

# Check if the mandatory arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Missing mandatory argument <config-file>."
  print_help
  return
fi

# Get the config file
config_file=$1
shift 1

# Select the config file folder based on the config file name
case "$config_file" in
  c1*)
    config_folder="train_c1_seeds"
    ;;
  c2*)
    config_folder="train_c2_seeds"
    ;;
  c3*)
    config_folder="train_c3_seeds"
    ;;
  c4*)
    config_folder="train_c4_seeds"
    ;;
  *)
    config_folder=""  # assume that config folder is already specified in policy
    ;;
esac

# Add the .py extension if not present
if [[ "$config_file" != *.py ]]; then
  config_file="${config_file}.py"
fi

# Check if the configuration file exists
if ! [ -f "config_files/$config_folder/$config_file" ]; then
  echo "Error: Configuration file 'config_files/$config_folder/$config_file' does not exist."
  return 1
fi

# Parse the arguments
config=""
config_folder=""
policy=""
policy_folder=""
policy_set=false
seed=0
seed_set=false

while [[ $# -gt 0 ]]; do
  case "$1" in

    -p|--policy)
      policy="$2"
      policy_set=true
      shift 2

      # Remove the .py extension if present
      if [[ "$policy" == *.py ]]; then
        policy="${policy%.py}"
      fi

      # Select the policy script_folder based on the policy name
      case "$policy" in
        c1*)
          policy_folder="train_c1"
          ;;
        c2*)
          echo "Warning: The selected policy '$policy' is a c2-type policy."
          policy_folder="train_c2"
          ;;
        c3*)
          policy_folder="train_c3"
          ;;
        c4*)
          echo "Warning: The selected policy '$policy' is a c4-type policy."
          policy_folder="train_c4"
          ;;
        *)
          policy_folder=""  # assume that script_folder is already specified in policy
          ;;
      esac

      # Check if the policy file exists and if it is compatible with the script
      if ! [ -d "results/$policy_folder/$policy" ]; then
        echo "Error: Policy script_folder 'results/$policy_folder/$policy' does not exist."
        return 1
      fi
      ;;

    -s|--seed)
      seed="$2"
      seed_set=true
      shift 2

      # Check if the seed is a positive integer
      if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "Error: Seed must be a positive integer."
        return 1
      fi

      # Raise warning for changing the seed
      echo "Warning: The seed from the config file is being changed to $seed"
      ;;

    -h|--help)
      print_help
      return
      ;;

    *)
      echo "Error: Unknown option '$1'."
      print_help
      return 1
      ;;

  esac
done

# Build command to execute
cmd=( taskset -c 0-7 python run_single_vehicle/train_dqn.py --mode train)

if $policy_set; then
  cmd+=(--policy "$policy_folder/$policy")
fi

if $seed_set; then
  cmd+=(--seed "$seed")
fi

# Run training
if $DEBUG ; then
  echo "Debug mode is ON."
  echo "Command: ${cmd[*]}"
  echo "Policy: $policy_folder/$policy"
fi

echo "Starting training..."
"${cmd[@]}"
echo -e "\nTraining completed."