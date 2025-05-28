#!/bin/bash

YAML_FILE="${1:-../arguments/args_demo.yaml}"
BATCH_SIZE=500
MAX_QUERIES=6550

echo "Using YAML file: $YAML_FILE"
echo "BATCH_SIZE: $BATCH_SIZE"
echo "MAX_QUERIES: $MAX_QUERIES"
echo "IMPORTANT! Please check that MAX_QUERIES is set to a number higher than the total number of query images in your input sequence."

START=0

while [[ "$START" -lt "$MAX_QUERIES" ]]; do
  echo "    ***** ***** *****"

  END=$((START + BATCH_SIZE - 1))

  if [[ "$END" -ge "$MAX_QUERIES" ]]; then
    END=$((MAX_QUERIES - 1))
    echo "END has been adjusted to $END."
  fi

  BATCH_RANGE="$START,$END"
  echo "BATCH START: $START , BATCH END: $END"
  pixi run -e lightglue batched --yaml_file "$YAML_FILE" --batch_range "$BATCH_RANGE"

  START=$((END + 1))

  sudo /sbin/swapoff -a && sudo /sbin/swapon -a

done

echo "Batch processing complete up to $MAX_QUERIES."

pixi run -e lightglue combine_batches --yaml_file "$YAML_FILE"