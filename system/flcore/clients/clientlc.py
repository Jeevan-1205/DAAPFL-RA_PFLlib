import torch
from flcore.clients.clientavg import clientAVG

class clientLC(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        # 1. Calculate local class distribution for FedLC
        self.sample_per_class = torch.zeros(self.num_classes).to(self.device)
        trainloader = self.load_train_data()
        
        # Determine the ignore_index if semantic segmentation is used
        ignore_index = getattr(self, "segmentation_ignore_index", -100)
        
        for x, y in trainloader:
            y = y.to(self.device)
            valid = y != ignore_index
            self.sample_per_class += torch.bincount(y[valid].flatten(), minlength=self.num_classes).float()
            
        # 2. Compute FedLC margin: tau * (n_y)^{-1/4}
        val = args.tau * (self.sample_per_class + 1e-8) ** (-0.25)
        self.calibration = val.to(self.device)
        
        # 3. Inject logit calibration directly into the loss function
        # This completely reuses the existing training loop in clientAVG.train()
        original_loss = self.loss
        def calibrated_loss(output, target):
            # Reshape calibration array to broadcast perfectly over logits [B, C, H, W, ...]
            cal_shape = [1, self.num_classes] + [1] * (output.dim() - 2)
            calibrated_output = output - self.calibration.view(cal_shape)
            return original_loss(calibrated_output, target)
            
        self.loss = calibrated_loss
