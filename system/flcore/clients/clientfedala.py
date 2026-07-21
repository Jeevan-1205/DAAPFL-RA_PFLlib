import copy
import torch
from flcore.clients.clientsegmentation import clientSegmentation
from utils.ALA import ALA
from utils.data_utils import read_client_data

class clientFedALA(clientSegmentation):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.eta = getattr(args, "eta", 1.0)
        self.rand_percent = getattr(args, "rand_percent", 80)
        self.layer_idx = getattr(args, "layer_idx", 0)
        self.ala_threshold = getattr(args, "ala_threshold", 0.1)
        
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        self.ALA = ALA(
            cid=self.id,
            loss=self.loss,
            train_data=train_data,
            batch_size=self.batch_size,
            rand_percent=self.rand_percent,
            layer_idx=self.layer_idx,
            eta=self.eta,
            device=self.device,
            threshold=self.ala_threshold
        )

    def set_parameters(self, model):
        """
        Overrides the standard parameter overwrite.
        Instead of simply copying the global model parameters to the local model,
        FedALA interpolates the global model and the previous local model 
        using the Adaptive Local Aggregation (ALA) module.
        """
        # The ALA module modifies self.model in-place
        self.ALA.adaptive_local_aggregation(model, self.model)
