# Project Migration & Setup Guide

This guide provides a comprehensive, step-by-step plan for migrating your PFLlib / DAAPFL-RA research project to a new machine and getting everything running smoothly from scratch.

> [!IMPORTANT]
> **What to Transfer**
> 1. Zip and transfer your entire `PFLlib` project directory.
> 2. **DO NOT** zip the `venv` folder or any `.conda` environment folders. They contain OS-specific binaries and absolute paths that will break on a new machine. We have explicitly added `venv/` to your `.gitignore` to prevent this.
> 3. If your `dataset/` (like xBD) is exceptionally large, consider transferring it separately or downloading it directly onto the new machine.

---

## 1. Environment Setup

Once you have extracted the project folder on your new machine, navigate into the project directory and run the automated setup script to recreate your exact Python environment.

```bash
cd /path/to/PFLlib

# 1. Ensure the script has execute permissions
chmod +x setup_env.sh

# 2. Run the setup script (this will create 'venv' and install requirements)
./setup_env.sh
```

> [!TIP]
> **Activating the Environment**
> Every time you start a new terminal session to work on this project, you must activate the virtual environment:
> ```bash
> source venv/bin/activate
> ```
> You will know it is active when your terminal prompt is prefixed with `(venv)`.

---

## 2. Verify Your Data

Ensure that your xBD dataset (or any other datasets you are using) is properly located in the `dataset/` directory according to your existing structure.

If you need to regenerate any synthetic dataset splits (e.g. for label skew experiments), use the generation scripts in the `dataset/` directory just like you did previously:
```bash
cd dataset
python generate_xBD.py noniid - dir  # (Adjust command based on your specific dataset scripts)
cd ..
```

---

## 3. Running Centralized Training

If you need to establish a centralized baseline (which you recently professionalized with tqdm, logging, and automated checkpointing), run your centralized training script:

```bash
cd system
python train_centralized.py
```
*Note: Make sure to pass any necessary arguments (like `--batch_size`, `--epochs`, `--lr`) if you configured `argparse` in that script.*

---

## 4. Running Federated Training (DAAPFL-RA)

To run your federated learning algorithms—including the newly implemented **DAAPFL-RA**—use the `main.py` entry point within the `system` directory. 

Here is an example command structure based on your recent work with semantic segmentation algorithms:

```bash
cd system

# Run the DAAPFL-RA algorithm
python main.py -algo DAAPFL-RA -data xBD -m segmentation -gr 100 -nc 10

# Run other baselines for comparison
python main.py -algo FedAvg -data xBD -m segmentation -gr 100 -nc 10
python main.py -algo FedProx -data xBD -m segmentation -gr 100 -nc 10
python main.py -algo pFedMe -data xBD -m segmentation -gr 100 -nc 10
```
> [!NOTE]
> - `-algo`: The algorithm name (e.g., DAAPFL-RA, FedAvg, FedProx, Ditto, SCAFFOLD).
> - `-data`: The dataset name (e.g., xBD).
> - `-m`: The model architecture (e.g., segmentation, Siamese U-Net).
> - `-gr`: Number of global rounds.
> - `-nc`: Number of clients.

---

## 5. Generating Analysis Reports

After your experiments finish running, the results (mIoU, Dice, Pixel Accuracy, F1-damage, etc.) will be saved to your CSV logs in the `results/` folder.

To synthesize these findings and compare the rare class collapse issues across algorithms, run your analysis script to generate the research-grade interactive HTML reports:

```bash
# Example command (adjust to your specific analysis script name)
python analysis/generate_report.py
```
This will output the `federated_analysis_report.html` file, which you can open in any web browser to view your bar charts and key metric summaries.

---

## Troubleshooting

> [!WARNING]
> **CUDA/PyTorch Mismatches**
> If you encounter issues where PyTorch does not detect your GPU on the new machine, it is likely because the new machine has a different CUDA version. If this happens:
> 1. Check your CUDA version: `nvcc --version`
> 2. Reinstall PyTorch specifically for that CUDA version from the [Official PyTorch Website](https://pytorch.org/get-started/locally/).
