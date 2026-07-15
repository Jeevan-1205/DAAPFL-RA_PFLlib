Centralized LODO

Resolution: 224×224
Loss: Dice(0.7) + Focal(0.3)
Scheduler: Cosine Warm Restarts
Momentum: 0.9
Class weights: Enabled
Epochs: 200

Command:
python train_centralized.py \
    --config config.yaml \
    --held_out_idx X \
    --epochs 200