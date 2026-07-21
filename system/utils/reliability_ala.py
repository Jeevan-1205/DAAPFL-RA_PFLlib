import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader
from typing import List, Tuple

class ReliabilityALA:
    def __init__(self,
                cid: int,
                loss: nn.Module,
                train_data: List[Tuple], 
                batch_size: int, 
                rand_percent: int, 
                layer_idx: int = 0,
                eta: float = 1.0,
                device: str = 'cpu', 
                threshold: float = 0.1,
                num_pre_loss: int = 10,
                rho: float = 0.1) -> None:
        """
        Initialize Reliability-aware ALA module

        Args:
            cid: Client ID. 
            loss: The loss function. 
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold.
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation.
            rho: Regularization coefficient for the Reliability-Guided Prior.
        """

        self.cid = cid
        self.loss = loss
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.rho = rho

        self.weights = None # Learnable local aggregation weights.
        self.start_phase = True

    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module,
                            reliability_prior: float) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. Uses an L2 prior to prevent meta-overfitting.

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 
            reliability_prior: The client's historical reliability memory (R_i_bar).
        """

        # randomly sample partial local training data (D_ALA)
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio * len(self.train_data))
        if rand_num == 0:
            return
            
        rand_idx = random.randint(0, len(self.train_data) - rand_num)
        rand_loader = DataLoader(self.train_data[rand_idx:rand_idx+rand_num], self.batch_size, drop_last=False)

        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        # deactivate ALA at the 1st communication iteration
        if torch.sum(params_g[0] - params[0]) == 0:
            return

        # preserve all the updates in the lower layers
        if self.layer_idx > 0:
            for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
                param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        if self.layer_idx > 0:
            params_p = params[-self.layer_idx:]
            params_gp = params_g[-self.layer_idx:]
            params_tp = params_t[-self.layer_idx:]
            
            # frozen the lower layers to reduce computational cost in Pytorch
            for param in params_t[:-self.layer_idx]:
                param.requires_grad = False
        else:
            params_p = params
            params_gp = params_g
            params_tp = params_t
            
        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_p]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
            param_t.data = param + (param_g - param) * weight

        # the mathematical prior target for W_i
        # If R_i_bar is 1 (highly reliable), target is 0 (keep local model)
        # If R_i_bar is 0 (highly unreliable), target is 1 (absorb global model)
        prior_target = 1.0 - reliability_prior

        # weight learning
        losses = []  # record losses
        cnt = 0  # weight training iteration counter
        while True:
            for x, y in rand_loader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                output = model_t(x)
                loss_value = self.loss(output, y) # local objective
                loss_value.backward()

                # update weight in this batch using gradients + L2 penalty towards prior
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    
                    # grad_W_task = param_t.grad * (param_g - param)
                    # grad_W_prior = rho * (weight - prior_target)
                    # W = W - eta * (grad_W_task + grad_W_prior)
                    grad_task = param_t.grad * (param_g - param)
                    grad_prior = self.rho * (weight - prior_target)
                    
                    weight.data = torch.clamp(weight - self.eta * (grad_task + grad_prior), 0, 1)

                # update temp local model in this batch
                for param_t, param, param_g, weight in zip(params_tp, params_p, params_gp, self.weights):
                    param_t.data = param + (param_g - param) * weight

            losses.append(loss_value.item())
            cnt += 1

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break

            # train the weight until convergence
            if len(losses) > self.num_pre_loss and np.std(losses[-self.num_pre_loss:]) < self.threshold:
                # print('Client:', self.cid, '\tStd:', np.std(losses[-self.num_pre_loss:]), '\tALA epochs:', cnt)
                break

        self.start_phase = False

        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
