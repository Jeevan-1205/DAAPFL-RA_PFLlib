#!/bin/bash

set -e

echo "======================================"
echo "Running FedPer (200 Rounds)"
echo "Started: $(date)"
echo "======================================"

python main.py \
    --config config.yaml \
    --algorithm FedPer \
    --global_rounds 200 \
    --local_epochs 3

echo ""
echo "======================================"
echo "FedPer completed!"
echo "Finished: $(date)"
echo "======================================"
