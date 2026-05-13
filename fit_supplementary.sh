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

echo "=== SR1 coefficient bootstrap ==="
$PYTHON -m "$MAIN" \
  --input "$FILE" \
  --sheet "$SHEET" \
  --model sr1 \
  $COMMON_ARGS \
  --coef-bootstrap \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --ci-level 0.95 \
  --table-format markdown

echo "=== Paired comparisons with SR1 ==="
$PYTHON -m "$MAIN" \
  --input "$FILE" \
  --sheet "$SHEET" \
  --model sr1 \
  $COMMON_ARGS \
  --paired-comparison \
  --compare-models absalom k kr krc krp krcs sr2 \
  --validation both \
  --bootstrap-n 10000 \
  --bootstrap-seed 12345 \
  --ci-level 0.95 \
  --paired-test signflip \
  --table-format markdown

echo "=== Cluster-level loss differences with SR1 ==="
$PYTHON -m "$MAIN" \
  --input "$FILE" \
  --sheet "$SHEET" \
  --model sr1 \
  $COMMON_ARGS \
  --cluster-loss \
  --compare-models absalom k kr krc krp krcs sr2 \
  --validation both \
  --table-format markdown
