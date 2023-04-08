#!/bin/bash

# Check if the required arguments are present
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <pid> <threshold>"
  exit 1
fi

# Store the arguments in variables
pid=$1
threshold=$2

while true; do
  # Get the temperature of the process
  temp=$(sensors | grep "Tctl" | awk '{print $2}' | cut -c 2-3)

  # Get the GPU temperature using nvidia-smi command
  gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader | tr -d '[:space:]')

  # Check if the temperature is above the threshold
  if (( $(echo "$temp > $threshold || $gpu_temp > $threshold" | bc -l) )); then
    echo "Temperature above threshold! Killing process $pid"
    kill "$pid"
    exit 0
  fi

  # Wait for 1 second before checking the temperature again
  sleep 1
done
