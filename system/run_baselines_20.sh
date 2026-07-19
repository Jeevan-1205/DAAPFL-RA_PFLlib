#!/bin/bash

set -e  # Stop if any command fails

echo "======================================"
echo "Starting 20-round baseline experiments"
echo "======================================"

# FedAvg
echo ""
echo "========== FedAvg =========="
python main.py \
    --config config.yaml \
    --algorithm FedAvg \
    --global_rounds 20 \
    --local_epochs 3

# FedProx
echo ""
echo "========== FedProx =========="
python main.py \
    --config config.yaml \
    --algorithm FedProx \
    --global_rounds 20 \
    --local_epochs 3

# Local
echo ""
echo "========== Local =========="
python main.py \
    --config config.yaml \
    --algorithm Local \
    --global_rounds 20 \
    --local_epochs 3

# FedPer
echo ""
echo "========== FedPer =========="
python main.py \
    --config config.yaml \
    --algorithm FedPer \
    --global_rounds 20 \
    --local_epochs 3

echo ""
echo "======================================"
echo "All experiments completed!"
echo "======================================"