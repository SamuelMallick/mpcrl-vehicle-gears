#!/bin/bash

DEBUG=false  # Set to true to enable debug mode

# Help message
print_help() {
  echo
  echo "Bash script to send notification to phone. Intended use is to send a notification after a batch of evaluations has been completed."
  echo "Usage: $0 <message> [options]"
  echo
  echo "Positional arguments:"
  echo "  message <str>          Message to send (required)"
  echo
  echo "Options:"
  echo "  -h, --help              Show this help message"
  echo
  echo "Example:"
  echo " $0 'Hello, World!'"
  echo
}

# Check if the mandatory arguments are provided
if [ "$#" -lt 1 ]; then
  echo "Missing text message."
  print_help
  return
else
  MESSAGE="$1"
  shift
fi

# Load .env variables
if [ ! -f .env ]; then
  echo "Error: .env file not found. Please create a .env file with the TOPIC and (optinally) TOKEN variables to be able to send ntfy notifications."
  return 1
fi
source .env

# Check that TOPIC is set
if [ -z "$TOPIC" ]; then
  echo "Error: TOPIC must be set in the .env file."
  return 1
fi

# Send notification using ntfy.sh
if [ -z "$TOKEN" ]; then
  echo "Warning: TOKEN is not set in the .env file. Notifications will be sent without authentication."
  curl -H "Title: Update from server" -d "$MESSAGE" https://ntfy.sh/$TOPIC
else
  curl -u ":$TOKEN" -H "Title: Update from server" -d "$MESSAGE" https://ntfy.sh/$TOPIC
fi
