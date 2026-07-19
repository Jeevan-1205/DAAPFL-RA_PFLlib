#!/bin/bash

set -e

echo "======================================"
echo "Running FedProx (200 Rounds)"
echo "Started: $(date)"
echo "======================================"

python main.py \
    --config config.yaml \
    --algorithm FedProx \
    --global_rounds 200 \
    --local_epochs 3

echo ""
echo "======================================"
echo "FedProx completed!"
echo "Finished: $(date)"
echo "======================================"
