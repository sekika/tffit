#!/bin/sh
PYTHON="python3"
MAIN="src.main"
FILE="supplementary.xlsx"
SHEET="Table S1"
DIGIT="3"
COMMON_ARGS="--digit $DIGIT --header-row 3 --data-start 5 --data-end 40"
for model in absalom k kr krc krp krcs sr1 sr2
do
  set -- --input "$FILE" --sheet "$SHEET" --model "$model" $COMMON_ARGS
  echo "=== $model: fit ==="
  $PYTHON -m "$MAIN" "$@"
  echo "=== $model: LOSO ==="
  $PYTHON -m "$MAIN" "$@" --cv loso
  echo "=== $model: LOYO ==="
  $PYTHON -m "$MAIN" "$@" --cv loyo
done