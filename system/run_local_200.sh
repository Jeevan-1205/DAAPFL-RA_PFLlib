#!/bin/bash

set -e

echo "======================================"
echo "Running Local (200 Rounds)"
echo "Started: $(date)"
echo "======================================"

python main.py \
    --config config.yaml \
    --algorithm Local \
    --global_rounds 200 \
    --local_epochs 3

echo ""
echo "======================================"
echo "Local completed!"
echo "Finished: $(date)"
echo "======================================"