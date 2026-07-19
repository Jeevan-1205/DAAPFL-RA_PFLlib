import time
import torch
import numpy as np
from flcore.clients.clientavg import clientAVG
from utils.segmentation_metrics import segmentation_confusion_matrix


class clientSegmentation(clientAVG):
    """FedAvg client with semantic-segmentation evaluation metrics."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.segmentation_ignore_index = getattr(args, "segmentation_ignore_index", None)

        if self.algorithm == "SCAFFOLD":
            self.client_c = []
            for param in self.model.parameters():
                self.client_c.append(torch.zeros_like(param))
            self.global_c = None
            self.global_model = None

            from flcore.optimizers.fedoptimizer import SCAFFOLDOptimizer
            self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learning_rate)
            
            self.learning_rate_scheduler = None
            if args.lr_schedule == "cosine":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.global_rounds)
            elif args.lr_schedule == "step":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
            elif args.lr_schedule == "cosine_warm_restarts":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=args.lr_t0, T_mult=args.lr_tmult)
            elif args.lr_schedule == "plateau":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=args.lr_plateau_factor, patience=args.lr_plateau_patience)

        elif self.algorithm == "Ditto":
            import copy
            self.model_per = copy.deepcopy(self.model)
            from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
            self.optimizer_per = PerturbedGradientDescent(
                self.model_per.parameters(), lr=self.learning_rate, mu=getattr(args, "mu", 0.0)
            )
            self.learning_rate_scheduler_per = None
            if args.lr_schedule == "cosine":
                self.learning_rate_scheduler_per = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_per, T_max=args.global_rounds)
            elif args.lr_schedule == "step":
                self.learning_rate_scheduler_per = torch.optim.lr_scheduler.StepLR(self.optimizer_per, step_size=args.lr_step_size, gamma=args.lr_gamma)
            elif args.lr_schedule == "cosine_warm_restarts":
                self.learning_rate_scheduler_per = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_per, T_0=args.lr_t0, T_mult=args.lr_tmult)
            elif args.lr_schedule == "plateau":
                self.learning_rate_scheduler_per = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_per, mode="max", factor=args.lr_plateau_factor, patience=args.lr_plateau_patience)

        elif self.algorithm == "pFedMe":
            import copy
            self.lamda = args.lamda
            self.K = args.K
            self.personalized_learning_rate = args.p_learning_rate

            self.local_params = copy.deepcopy(list(self.model.parameters()))
            self.personalized_params = copy.deepcopy(list(self.model.parameters()))

            from flcore.optimizers.fedoptimizer import pFedMeOptimizer
            self.optimizer = pFedMeOptimizer(
                self.model.parameters(), lr=self.personalized_learning_rate, lamda=self.lamda
            )
            
            self.learning_rate_scheduler = None
            if args.lr_schedule == "cosine":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.global_rounds)
            elif args.lr_schedule == "step":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
            elif args.lr_schedule == "cosine_warm_restarts":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=args.lr_t0, T_mult=args.lr_tmult)
            elif args.lr_schedule == "plateau":
                self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="max", factor=args.lr_plateau_factor, patience=args.lr_plateau_patience)

    def set_parameters(self, model, global_c=None):
        if self.algorithm == "SCAFFOLD":
            self.global_c = global_c
            self.global_model = model

        # FedRep uses the same encoder-only loading as FedPer
        if self.algorithm in ("FedPer", "FedRep"):
            target_global = model.base if hasattr(model, "base") else model
            target_local = self.model.base if hasattr(self.model, "base") else self.model

            if hasattr(target_local, "load_shared_encoder") and hasattr(target_global, "shared_encoder_state"):
                target_local.load_shared_encoder(target_global.shared_encoder_state())
            elif hasattr(target_local, "encoder") and hasattr(target_global, "encoder"):
                for new_param, old_param in zip(target_global.encoder.parameters(), target_local.encoder.parameters()):
                    old_param.data = new_param.data.clone()
            else:
                super().set_parameters(model)
        else:
            super().set_parameters(model)

    # ------------------------------------------------------------------ #
    #  FedRep two-stage training                                          #
    # ------------------------------------------------------------------ #

    def train(self):
        if self.algorithm == "Ditto":
            self._train_ditto()
        elif self.algorithm == "FedRep":
            self._train_fedrep()
        elif self.algorithm == "SCAFFOLD":
            self._train_scaffold()
        elif self.algorithm == "pFedMe":
            self._train_pfedme()
        else:
            super().train()  # FedAvg / FedProx / FedPer — untouched

    def _train_ditto(self):
        trainloader = self.load_train_data()
        self.model.train()
        self.model_per.train()
        start_time = time.time()

        # 1. Train personalized model (ptrain)
        head_epochs = getattr(self.args, "plocal_epochs", 1)
        print(f"[Ditto] Client {self.id} — Stage 1: training personalized model ({head_epochs} epoch(s))")
        for epoch in range(head_epochs):
            self.init_class_distribution_tracker()
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                self.update_class_distribution_tracker(y)
                output = self.model_per(x)
                loss = self.loss(output, y)
                self.optimizer_per.zero_grad()
                loss.backward()
                self.optimizer_per.step(self.model.parameters(), self.device)
            self.log_class_distribution_summary(epoch)

        # 2. Train global model (train)
        print(f"[Ditto] Client {self.id} — Stage 2: training global model ({self.local_epochs} epoch(s))")
        for epoch in range(self.local_epochs):
            self.init_class_distribution_tracker()
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                self.update_class_distribution_tracker(y)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.log_class_distribution_summary(epoch)

        if self.learning_rate_scheduler is not None and self.args.lr_schedule != "plateau":
            self.learning_rate_scheduler.step()
        if hasattr(self, "learning_rate_scheduler_per") and self.learning_rate_scheduler_per is not None and self.args.lr_schedule != "plateau":
            self.learning_rate_scheduler_per.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _train_fedrep(self):
        """FedRep alternating local optimisation (Collins et al., 2021).

        Stage 1 (head_epochs):
            Freeze encoder.  Train decoder + segmentation head only.

        Stage 2 (encoder_epochs):
            Freeze decoder + head.  Train encoder only.
            
        To ensure fair computation comparison with FedAvg/FedPer, total
        dataset passes per round is strictly bounded by local_epochs:
            head_epochs + encoder_epochs == local_epochs
            
        We use the persistent `self.optimizer` for both stages. PyTorch's
        optimizer ignores parameters with `requires_grad=False` (since
        `p.grad is None`), naturally preserving momentum state for the 
        frozen layers until their respective training stage.
        """
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()

        # Compute fair epoch split
        head_epochs = getattr(self.args, "plocal_epochs", 1)
        encoder_epochs = self.local_epochs - head_epochs
        
        # Guard against invalid configs (e.g., local_epochs=1, plocal_epochs=1)
        if encoder_epochs < 0:
            head_epochs = self.local_epochs
            encoder_epochs = 0
            print(f"[FedRep] WARNING: local_epochs ({self.local_epochs}) < "
                  f"plocal_epochs ({getattr(self.args, 'plocal_epochs', 1)}). "
                  "Encoder will not train!")

        # Resolve the actual SiameseUNet (may be wrapped in BaseHeadSplit)
        model = self.model.base if hasattr(self.model, "base") else self.model

        enc_params  = list(model.encoder.parameters())
        pers_params = list(model.decoder.parameters()) + list(model.fc.parameters())

        # ---- Stage 1: freeze encoder, train personalized decoder + head ----
        for p in enc_params:
            p.requires_grad = False
        for p in pers_params:
            p.requires_grad = True

        print(f"[FedRep] Client {self.id} — Stage 1: training decoder+head "
              f"({head_epochs} epoch(s))")
        for epoch in range(head_epochs):
            self.init_class_distribution_tracker()
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                self.update_class_distribution_tracker(y)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.log_class_distribution_summary(epoch)

        # ---- Stage 2: freeze decoder + head, train encoder ----
        for p in enc_params:
            p.requires_grad = True
        for p in pers_params:
            p.requires_grad = False

        print(f"[FedRep] Client {self.id} — Stage 2: training encoder "
              f"({encoder_epochs} epoch(s))")
        for epoch in range(encoder_epochs):
            self.init_class_distribution_tracker()
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                self.update_class_distribution_tracker(y)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.log_class_distribution_summary(epoch)

        # Restore all params to requires_grad=True for eval / set_parameters
        for p in self.model.parameters():
            p.requires_grad = True

        if self.learning_rate_scheduler is not None and self.args.lr_schedule != "plateau":
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def _train_scaffold(self):
        trainloader = self.load_train_data()
        self.model.train()
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        print(f"[SCAFFOLD] Client {self.id} — training ({max_local_epochs} epoch(s))")
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
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.global_c, self.client_c)
            self.log_class_distribution_summary(epoch)

        self.num_batches = len(trainloader)
        self.update_yc(max_local_epochs)

        if self.learning_rate_scheduler is not None and self.args.lr_schedule != "plateau":
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def update_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(), self.model.parameters()):
            ci.data = ci - c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs=None):
        if max_local_epochs is None:
            max_local_epochs = self.local_epochs
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1/self.num_batches/max_local_epochs/self.learning_rate * (x - yi))

        return delta_y, delta_c

    def _train_pfedme(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        print(f"[pFedMe] Client {self.id} — training ({max_local_epochs} epoch(s))")
        for epoch in range(max_local_epochs):
            self.init_class_distribution_tracker()
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                self.update_class_distribution_tracker(y)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # K is number of personalized steps
                for i in range(self.K):
                    output = self.model(x)
                    loss = self.loss(output, y)
                    self.optimizer.zero_grad()
                    loss.backward()
                    # finding aproximate theta
                    self.personalized_params = self.optimizer.step(self.local_params, self.device)

                # update local weight after finding aproximate theta
                for new_param, localweight in zip(self.personalized_params, self.local_params):
                    localweight = localweight.to(self.device)
                    localweight.data = localweight.data - self.lamda * self.learning_rate * (localweight.data - new_param.data)

            self.log_class_distribution_summary(epoch)

        if self.learning_rate_scheduler is not None and self.args.lr_schedule != "plateau":
            self.learning_rate_scheduler.step()

        self.update_parameters(self.model, self.local_params)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        
        if self.algorithm == "Ditto":
            eval_model = self.model_per
        elif self.algorithm == "pFedMe":
            import copy
            original_params = copy.deepcopy(list(self.model.parameters()))
            self.update_parameters(self.model, self.personalized_params)
            eval_model = self.model
        else:
            eval_model = self.model
            
        eval_model.eval()

        confusion = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.int64,
        )
        print(f"Client {self.id} evaluation started")
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                output = eval_model(x)
                pred = torch.argmax(output, dim=1)
                if pred.shape != y.shape:
                    raise ValueError(
                        "segmentation predictions and masks must have matching "
                        f"shape, got {tuple(pred.shape)} and {tuple(y.shape)}."
                    )

                confusion += segmentation_confusion_matrix(
                    pred,
                    y,
                    self.num_classes,
                    ignore_index=self.segmentation_ignore_index,
                )
            print(f"Client {self.id} evaluation finished")

        if self.algorithm == "pFedMe":
            self.update_parameters(self.model, original_params)

        return confusion, int(confusion.sum().item())
