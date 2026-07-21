# DAAPFL-RA: Damage Assessment Adaptive Personalized Federated Learning

A highly specialized federated semantic segmentation framework built on a tailored version of [PFLlib](https://github.com/TsingZ0/PFLlib). This repository is designed to tackle rare class collapse in building damage assessment, specifically using the **xBD** dataset.

## 🎯 Key Features & Contributions

- **DAAPFL-RA Algorithm:** Implements Reliability-Aware Adaptive Personalized Federated Learning to dynamically aggregate client models based on local data exposure, mitigating gradient collapse for rare damage classes (e.g., "Minor" and "Major" damage).
- **Siamese U-Net Backbone:** Employs a dual-encoder Siamese U-Net architecture optimized for high-resolution satellite imagery segmentation.
- **Reliability-Aware Aggregation:** Utilizes `ReliabilityALA` to scale aggregation weights based on empirical class exposure metrics, ensuring stable convergence across highly heterogeneous (non-IID) client data distributions.
- **Extensive Baselines:** Contains clean, integrated implementations of top-tier federated learning algorithms for rigorous performance comparison. Supported baselines include:
  - `FedAvg`, `FedProx`, `FedPer`, `FedRep`, `pFedMe`, `Ditto`, `SCAFFOLD`, `FedLC`, `Local`, `FedALA`

---

## 🛠️ Environment Setup

We provide an automated setup script to instantly recreate the Python environment required to run this project.

1. **Clone or Extract the Repository:**
   Navigate into the project root directory.
2. **Run the Setup Script:**
   ```bash
   chmod +x setup_env.sh
   ./setup_env.sh
   ```
3. **Activate the Environment:**
   You must activate the virtual environment every time you open a new terminal:
   ```bash
   source venv/bin/activate
   ```

---

## 📊 Dataset Preparation

This framework relies on the **xBD** dataset for disaster damage assessment. 

Ensure that your xBD dataset is properly located in the `dataset/xBD/` directory. To generate non-IID federated data splits across your clients (e.g., using Dirichlet distribution for practical label skew), use the provided dataset generation script:

```bash
cd dataset
python generate_xBD.py noniid - dir
```

---

## 🚀 Usage & Execution

All experiments are executed from the `system/` directory.

### Centralized Baseline
To establish a baseline using standard centralized training, we provide a professionalized training script featuring progress tracking (tqdm) and automated checkpointing:

```bash
cd system
python train_centralized.py
```

### Federated Training (DAAPFL-RA)
To run your federated learning algorithms, use the `main.py` entry point. 

**Run DAAPFL-RA:**
```bash
cd system
python main.py -algo DAAPFL-RA -data xBD -m segmentation -gr 100 -nc 10
```

**Run Baseline Comparisons:**
```bash
python main.py -algo FedAvg -data xBD -m segmentation -gr 100 -nc 10
python main.py -algo FedProx -data xBD -m segmentation -gr 100 -nc 10
python main.py -algo pFedMe -data xBD -m segmentation -gr 100 -nc 10
```

**Key Arguments:**
- `-algo`: Algorithm name (e.g., `DAAPFL-RA`, `FedAvg`, `SCAFFOLD`, `Ditto`).
- `-data`: Dataset name (default: `xBD`).
- `-m`: Model architecture (use `segmentation` for the Siamese U-Net).
- `-gr`: Number of global federated rounds.
- `-nc`: Total number of simulated clients.

---

## 📈 Results & Analysis

After experiments complete, detailed round-level metrics (mIoU, Dice, Pixel Accuracy, F1-damage) are automatically saved as CSV logs in the `results/` directory.

To synthesize these findings into an interactive, research-grade HTML report that visualizes the rare class performance gap, run the analysis script:

```bash
# Depending on your exact analysis script name
python analysis/generate_report.py
```
This outputs `federated_analysis_report.html`, which you can view in any standard web browser.

---

## 🙏 Acknowledgments

This specialized codebase was originally forked and heavily refactored from the [PFLlib: Personalized Federated Learning Library and Benchmark](https://github.com/TsingZ0/PFLlib). We thank the original authors for providing a robust foundation for federated learning research.
