import copy
import torch
import torch.nn as nn
from losses import build_loss
from torch.utils.data import WeightedRandomSampler
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = build_loss(
            name="dice_focal",
            num_classes=args.num_classes,
            alpha=args.class_weights,
            dice_weight=args.dice_weight,
            focal_weight=args.focal_weight,
        )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=args.momentum,
        )
        print(f"[Client {self.id}] "
                f"LR={self.optimizer.param_groups[0]['lr']:.6f} "
                f"Momentum={self.optimizer.param_groups[0]['momentum']}")

        self.learning_rate_scheduler = None

        if args.lr_schedule == "cosine":

            self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.global_rounds,
            )

        elif args.lr_schedule == "step":

            self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=args.lr_step_size,
                gamma=args.lr_gamma,
            )

        elif args.lr_schedule == "cosine_warm_restarts":

            self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=args.lr_t0,
                T_mult=args.lr_tmult,
            )

        elif args.lr_schedule == "plateau":

            self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=args.lr_plateau_factor,
                patience=args.lr_plateau_patience,
            )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        train_data = read_client_data(
            self.dataset,
            self.id,
            is_train=True,
            few_shot=self.few_shot,
        )

        # Only use weighted sampling for xBD datasets that support has_damage()
        if hasattr(train_data, "has_damage"):

            weights = []

            damage_tiles = 0
            background_tiles = 0

            for i in range(len(train_data)):
                if train_data.has_damage(i):
                    damage_tiles += 1
                    weights.append(2.0)      # oversample damage
                else:
                    background_tiles += 1
                    weights.append(1.0)

            print(
                f"Client {self.id}: "
                f"{damage_tiles} damage / "
                f"{background_tiles} background tiles"
            )

            sampler = WeightedRandomSampler(
                weights,
                num_samples=len(train_data),
                replacement=True,
            )

            return DataLoader(
                train_data,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )

        # Default loader for non-segmentation datasets
        return DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
        )
        
    def set_parameters(self, model):
        self.model.load_state_dict(model.state_dict(), strict=True)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
