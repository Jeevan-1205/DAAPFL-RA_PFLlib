import copy
import time
import torch
import numpy as np
from flcore.servers.serversegmentation import ServerSegmentation
from flcore.clients.clientdaapfl import clientDAAPFL

class ServerDAAPFL(ServerSegmentation):
    def __init__(self, args, times):
        super().__init__(args, times)
        
        # Clear default clients created by ServerSegmentation
        self.clients = []
        self.held_out_client = None
        
        # Override clients with clientDAAPFL
        self.set_clients(clientDAAPFL)
        print("\n========== CLIENT LIST (DAAPFL) ==========")
        for c in self.clients:
            print(f"Client ID: {c.id}")
        print("==========================================\n")

        self.beta_ema = getattr(args, "beta_ema", 0.9)
        self.omega = getattr(args, "omega", 0.5)
        self.tau_agg = getattr(args, "tau_agg", 0.5)
        
        # Initialize EMA Reliability Memory (R_i_bar)
        self.reliability_memory = {c.id: 0.5 for c in self.clients}
        self.v_g = None # Global momentum

    def send_models(self):
        # We need to set the R_i_bar for each client before calling set_parameters
        for client in self.clients:
            client.R_i_bar = self.reliability_memory[client.id]
        super().send_models()

    def aggregate_parameters(self):
        assert len(self.uploaded_ids) > 0
        
        # 1. Compute Optimization Alignment (A_i) and update Reliability
        A_i = {}
        current_R_i = {}
        
        print("\n" + "-"*80)
        print(f"DAAPFL-RA Reliability Tracking (Round)")
        print("-" * 80)
        print(f"{'Client':<8} | {'G_i':<8} | {'A_i':<8} | {'R_i (cur)':<10} | {'R_i_bar (EMA)':<14}")
        
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            # Compute Delta_theta_i
            delta_i = []
            for param_g, param_i in zip(self.global_model.parameters(), client_model.parameters()):
                delta_i.append((param_i.data - param_g.data).view(-1))
            delta_i_vec = torch.cat(delta_i)
            
            if self.v_g is None:
                # Round 1 initialization: neutral alignment
                A_i[cid] = 0.5
            else:
                norm_delta = torch.norm(delta_i_vec, p=2)
                norm_vg = torch.norm(self.v_g, p=2)
                if norm_delta == 0 or norm_vg == 0:
                    cos_sim = 0.0
                else:
                    cos_sim = torch.dot(delta_i_vec, self.v_g) / (norm_delta * norm_vg)
                A_i[cid] = 0.5 * (1.0 + cos_sim.item())
                
            # Retrieve G_i from the client object (updated during local training)
            client = next(c for c in self.clients if c.id == cid)
            G_i = client.G_i
            
            # Weighted Fusion
            R_i = self.omega * G_i + (1.0 - self.omega) * A_i[cid]
            current_R_i[cid] = R_i
            
            # Update EMA Memory
            self.reliability_memory[cid] = self.beta_ema * self.reliability_memory[cid] + (1.0 - self.beta_ema) * R_i
            
            print(f"{cid:<8} | {G_i:<8.4f} | {A_i[cid]:<8.4f} | {R_i:<10.4f} | {self.reliability_memory[cid]:<14.4f}")
        print("-" * 80)
        
        # 2. Temperature-scaled Softmax Aggregation
        w_unnormalized = []
        for cid in self.uploaded_ids:
            client = next(c for c in self.clients if c.id == cid)
            n_i = client.train_samples
            R_i_bar = self.reliability_memory[cid]
            val = n_i * np.exp(R_i_bar / self.tau_agg)
            w_unnormalized.append(val)
            
        w_sum = sum(w_unnormalized)
        w_normalized = [w / w_sum for w in w_unnormalized]
        
        # Print aggregation weights
        print("\nAggregation Weights:")
        for cid, w_n in zip(self.uploaded_ids, w_normalized):
            print(f"Client {cid}: {w_n:.4f}")
        print("-" * 80)
        
        # Save previous global model parameters to compute v_g for next round
        prev_global_params = [param.data.clone() for param in self.global_model.parameters()]
        
        # Execute parameter aggregation
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(w_normalized, self.uploaded_models):
            for target_param, source_param in zip(self.global_model.parameters(), client_model.parameters()):
                target_param.data += source_param.data.clone() * w
                
        # 3. Update global momentum v_g
        delta_g = []
        for prev_param, curr_param in zip(prev_global_params, self.global_model.parameters()):
            delta_g.append((curr_param.data - prev_param.data).view(-1))
        self.v_g = torch.cat(delta_g)
