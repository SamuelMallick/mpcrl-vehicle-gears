#!/bin/bash

DEBUG=false  # Set to true to enable debug mode

# Help message
print_help() {
  echo
  echo "Bash script to run a batch of evaluations experiments."
  echo "Usage: $0 <type> <script_name> [options]"
  echo
  echo "Positional arguments:"
  echo "  type <str>              Type of evaluation to run {single, platoon}"
  echo "  script_name <str>       Script name to run (required)"
  echo
  echo "Options:"
  echo "  -c, --config <file>     Configuration file to use (must be under config_files/eval_seeds/)"
  echo "  -p, --policy <str>      Policy file to use (takes effect only if script_name is l_mpc)"
  echo "  -s, --seed <int>        Start seed value"
  echo "  -n, --n-evals <int>     Number of evaluations to run"
  echo "  -e, --seed-end <int>    End seed value (alternative to --n-evals)"
  echo "  -t, --max-time <float>  Maximum time for knitro and gurobi"
  echo "  -h, --help              Show this help message"
  echo "Note: the default values are declared in parse_config.py."
  echo
  echo "Example:"
  echo " $0 single l_mpc.py --policy c4_seed5 --seed 101 --n-evals 25 "
  echo
}

# Check if the mandatory arguments are provided
if [ "$#" -lt 2 ]; then
  echo "Missing one or more mandatory arguments."
  print_help
  return
fi

# Get the argument values
script_type=$1
script_name=$2
shift 2

# Validate the script type
if ! [[ "$script_type" =~ ^(single|platoon)$ ]]; then
  echo "Error: Invalid evaluation type '$script_type'. Must be one of {single, platoon}."  # Other experiment types will be added here
  print_help
  return
fi

# Remove the .py extension if present
if [[ "$script_name" != *.py ]]; then
  script_name="${script_name}.py"
fi 

# Select the script folder based on the script type
case "$script_type" in
  single)
    script_folder="run_single_vehicle"
    if ! [ -f "$script_folder/$script_name" ]; then
      echo "Error: File '$script_folder/$script_name' does not exist."
      return
    fi
    ;;
  platoon)
    script_folder="run_platoon"
    if ! [ -f "$script_folder/$script_name" ]; then
      echo "Error: File '$script_folder/$script_name' does not exist."
      return
    fi
    ;;
esac

# Parse optional arguments
config=""
config_set=false
policy=""
policy_set=false
max_time=0
max_time_set=false
seed_start=0
seed_set=false
seed_end=0

while [[ $# -gt 0 ]]; do
  case "$1" in

    -c|--config)
      config="$2"
      config_set=true
      shift 2

      # Add the .py extension if not present
      if [[ "$config" != *.py ]]; then
        config="${config}.py"
      fi

      # Append eval_seeds/ to the config path
      config="eval_seeds/$config"

      # Check if the configuration file exists
      if ! [ -f "config_files/$config" ]; then
        echo "Error: Configuration file 'config_files/$config' does not exist."
        return 1
      fi
      ;;

    -p|--policy)
      policy="$2"
      policy_set=true
      shift 2

      # Select the policy script_folder based on the policy name
      case "$policy" in
        c1*)
          policy_folder="train_c1"
          ;;
        c2*)
          policy_folder="train_c2"
          ;;
        c3*)
          policy_folder="train_c3"
          ;;
        c4*)
          policy_folder="train_c4"
          ;;
        *)
          policy_folder=""  # assume that script_folder is already specified in policy
          ;;
      esac

      # Remove the .py extension if present
      if [[ "$policy" == *.py ]]; then
        policy="${policy%.py}"
      fi

      # Check if the policy file exists and if it is compatible with the script
      if ! [ -d "results/$policy_folder/$policy" ]; then
        echo "Error: Policy script_folder 'results/$policy_folder/$policy' does not exist."
        return 1
      fi
      if [[ "$script_name" != "l_mpc.py" && -n "$policy" ]]; then
        echo "Warning: Policy definition only takes effect with l_mpc.py script."
      fi
      ;;

    -s|--seed)
      seed_start="$2"
      seed_end="$seed_start"
      seed_set=true
      shift 2

      # Check if the seed is a positive integer
      if ! [[ "$seed_start" =~ ^[0-9]+$ ]]; then
        echo "Error: Seed must be a positive integer."
        return 1
      fi
      ;;

    -n|--n-evals)
      seed_end=$(("$seed_start" + "$2" - 1))
      shift 2

      # Check if the number of evaluations is a positive integer
      if ! [[ "$seed_start" =~ ^[0-9]+$ ]]; then
        echo "Error: Number of evaluations must be a positive integer."
        return 1
      fi
      ;;

    -e|--seed-end)
      seed_end="$2"
      shift 2

      # Check if the seed end is a positive integer
      if ! [[ "$seed_end" =~ ^[0-9]+$ ]]; then
        echo "Error: Seed end must be a positive integer."
        return 1
      fi
      ;;

    -t|--max-time)
      max_time="$2"
      max_time_set=true
      shift 2

      # Check if the max time is a positive float
      if ! [[ "$max_time" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        echo "Error: Max time must be a positive float."
        return 1
      fi
      ;;

    -h|--help)
      print_help
      return 0
      ;;

    *)
      echo "Unknown option: $1"
      print_help
      return 1
      ;;
      
  esac
done

# Build command to execute
cmd=( taskset -c 0-7 python "$script_folder/$script_name" --mode eval )

if $config_set; then
  cmd+=(--config "$config")
fi

if $policy_set; then
  cmd+=(--policy "$policy_folder/$policy")
fi

if $max_time_set; then
  cmd+=(--max-time "$max_time")
fi

# Run evaluations
if $DEBUG ; then
  echo "Debug mode is ON."
  echo "Command: ${cmd[*]}"
  echo "Evaluation type: $script_type"
  echo "Script name: $script_name"
  echo "Script folder: $script_folder"
  echo "Configuration: $config"
  echo "Policy: $policy"
  echo "Max time: $max_time"
  echo "Seed start: $seed_start"
  echo "Seed end: $seed_end"
fi

echo "Running $script_type evaluation of: $script_folder/$script_name"
if $seed_set; then
  echo "Using seeds from $seed_start to $seed_end"
  for seed in $( eval echo {$seed_start..$seed_end} ); do
    echo -e "\n########### SEED $seed ###########"
    "${cmd[@]}" --seed "$seed"
  done
else
  "${cmd[@]}"
fi
echo -e "\nEvaluation completed."
