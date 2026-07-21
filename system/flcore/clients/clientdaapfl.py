import time
import torch
import copy
import numpy as np
from torch.utils.data import DataLoader, Subset
from flcore.clients.clientsegmentation import clientSegmentation
from utils.reliability_ala import ReliabilityALA
from utils.data_utils import read_client_data
from utils.segmentation_metrics import segmentation_metrics, segmentation_confusion_matrix

class clientDAAPFL(clientSegmentation):
    """DAAPFL-RA client."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.eta = getattr(args, "eta", 1.0)
        self.rand_percent = getattr(args, "rand_percent", 80)
        self.layer_idx = getattr(args, "layer_idx", 0)
        self.ala_threshold = getattr(args, "ala_threshold", 0.1)
        
        self.rho = getattr(args, "rho", 0.1)
        self.lambda_max = getattr(args, "lambda_max", 1.0)
        self.gamma = getattr(args, "gamma", 5.0)
        self.kappa = getattr(args, "kappa", 1.0)
        
        self.R_i_bar = 0.5
        
        # Load full local training data
        full_train_data = read_client_data(self.dataset, self.id, is_train=True)
        
        # Partition into 80% train and 20% val
        dataset_len = len(full_train_data)
        indices = list(range(dataset_len))
        np.random.RandomState(42).shuffle(indices)
        split = int(0.8 * dataset_len)
        
        train_indices = indices[:split]
        val_indices = indices[split:]
        
        self.train_data = Subset(full_train_data, train_indices)
        self.val_data = Subset(full_train_data, val_indices)
        
        self.ALA = ReliabilityALA(
            cid=self.id,
            loss=self.loss,
            train_data=self.train_data,
            batch_size=self.batch_size,
            rand_percent=self.rand_percent,
            layer_idx=self.layer_idx,
            eta=self.eta,
            device=self.device,
            threshold=self.ala_threshold,
            rho=self.rho
        )
        
        self.received_global_model = None
        self.G_i = 0.5

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.train_data, batch_size, drop_last=False, shuffle=True)
        
    def load_val_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.val_data, batch_size, drop_last=False, shuffle=False)

    def set_parameters(self, model):
        # Save a copy of the global model for evaluating G_i and for the proximal term
        self.received_global_model = copy.deepcopy(model)
        # Perform Reliability-Guided ALA. This updates self.model in-place.
        self.ALA.adaptive_local_aggregation(model, self.model, self.R_i_bar)

    def compute_f1_dam(self, eval_model, dataloader):
        eval_model.eval()
        confusion = torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)
        with torch.no_grad():
            for x, y in dataloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                output = eval_model(x)
                pred = torch.argmax(output, dim=1)
                confusion += segmentation_confusion_matrix(
                    pred, y, self.num_classes, ignore_index=self.segmentation_ignore_index
                )
        metrics = segmentation_metrics(confusion)
        return metrics["f1_dam"]

    def train(self):
        trainloader = self.load_train_data()
        valloader = self.load_val_data()
        start_time = time.time()
        
        # 1. Evaluate global model on val_data
        metric_global = self.compute_f1_dam(self.received_global_model, valloader)
        
        self.model.train()
        
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        # Compute proximal coefficient alpha_i
        alpha_i = self.lambda_max * np.exp(-self.gamma * self.R_i_bar)

        for epoch in range(max_local_epochs):
            self.init_class_distribution_tracker()
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                self.update_class_distribution_tracker(y)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                output = self.model(x)
                loss = self.loss(output, y)
                
                # Add dynamic proximal regularization term
                proximal_term = 0.0
                for w, w_t in zip(self.model.parameters(), self.received_global_model.parameters()):
                    proximal_term += (w - w_t).norm(2) ** 2
                loss = loss + (alpha_i / 2) * proximal_term

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            self.log_class_distribution_summary(epoch)

        if self.learning_rate_scheduler is not None and self.args.lr_schedule != "plateau":
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        # 3. Evaluate updated local model on val_data
        metric_local = self.compute_f1_dam(self.model, valloader)
        
        # Compute Validation Generalization Gain
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
            
        self.G_i = sigmoid(self.kappa * (metric_local - metric_global))
        
        # To avoid NaNs if metric difference is huge
        if np.isnan(self.G_i):
            self.G_i = 0.5
            
        print(f"[DAAPFL-RA] Client {self.id} | F1_global: {metric_global:.4f} | F1_local: {metric_local:.4f} | G_i: {self.G_i:.4f} | alpha_i: {alpha_i:.4f}")
