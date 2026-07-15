import os
import time
import numpy as np
import torch
import yaml
from tqdm import tqdm
from datetime import datetime
from flcore.clients.clientsegmentation import clientSegmentation
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from utils.segmentation_metrics import segmentation_metrics
from utils.csv_logger import CSVLogger, make_run_id



class ServerSegmentation(Server):
    """Server with semantic-segmentation evaluation and result saving.

    Communication, client sampling, model aggregation, and checkpointing are
    inherited unchanged from ``Server``.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # create segmentation clients
        self.set_clients(clientSegmentation)
        print("\n========== CLIENT LIST ==========")
        for c in self.clients:
            print(f"Client ID: {c.id}")
        print("=================================\n")

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.rs_test_pixel_acc = []
        self.rs_test_dice = []
        self.rs_test_dice_per_class = []
        self.rs_test_iou = []
        self.rs_test_miou = []
        self.rs_test_f1_dam = []

        # CSV logging
        self.run_id = make_run_id()
        self.round_logger = None
        self._last_eval_metrics = {}

    def set_clients(self, clientObj=clientSegmentation):
        super().set_clients(clientObj)

    def set_new_clients(self, clientObj=clientSegmentation):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args,
                            id=i,
                            train_samples=len(train_data),
                            test_samples=len(test_data),
                            train_slow=False,
                            send_slow=False)
            self.new_clients.append(client)

    def _get_result_path(self):
        """Shared by save_results() and train() so both agree on the folder."""
        result_path = "results"

        if getattr(self.args, "output_dir", None):
            result_path = os.path.join(result_path, self.args.output_dir)

        os.makedirs(result_path, exist_ok=True)
        return result_path

    def _get_run_name(self):
        """Descriptive, filesystem-safe name identifying this specific run.

        e.g. FedAvg_xBD_lodo_fold0_gr200_20260707_153045
        Cached after first call so it stays identical across train()/save_results().
        """
        if getattr(self, "_run_name", None):
            return self._run_name

        parts = [str(self.algorithm), str(self.dataset)]

        protocol = getattr(self, "protocol", None)
        if protocol == "lodo" and self.held_out_idx is not None:
            parts.append("lodo")
            parts.append(f"fold{self.held_out_idx}")
        elif protocol:
            parts.append(str(protocol))

        parts.append(f"gr{self.global_rounds}")
        parts.append(self.run_id)

        self._run_name = "_".join(parts)
        return self._run_name

    def _init_round_logger(self, result_path):
        round_csv_name = f"{self._get_run_name()}_rounds.csv"
        round_csv_path = os.path.join(result_path, round_csv_name)

        self.round_logger = CSVLogger(
            round_csv_path,
            fieldnames=[
                "round", "held_out_idx", "evaluated",
                "pixel_acc", "dice", "miou", "iou_mean","f1_dam",
                "std_pixel_acc", "std_miou", "train_loss",
                "round_time_sec",
            ],
        )
        print(f"Logging per-round metrics to: {round_csv_path}")

    def train(self):
        result_path = self._get_result_path()
        self._init_round_logger(result_path)

        print("\n" + "═"*60)
        print("              FedAvg Training")
        print("═"*60)
        print(f"Algorithm      : {self.algorithm}")
        print(f"Dataset        : {self.dataset}")
        print(f"Protocol       : {self.protocol}")
        print(f"Clients        : {self.num_clients}")
        print(f"Global Rounds  : {self.global_rounds}")
        print(f"Local Epochs   : {self.local_epochs}")
        print(f"Join Ratio     : {self.join_ratio}")
        print("═"*60 + "\n")

        print("\n============= Initial Global Model =============")

        self.send_models()

        self.evaluate()
        
        print("="*60)



        round_bar = tqdm(
            range(self.global_rounds),
            desc=f"{self.algorithm} ({self.protocol})",
            unit="round",
            ncols=120,
        )

        for i in round_bar:
            evaluated_this_round = False

            s_t = time.time()

            self.selected_clients = self.select_clients()

            # Send latest global model
            self.send_models()

            for client in self.selected_clients:
                client.train()


            self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)

            self.aggregate_parameters()

            # Broadcast aggregated model before evaluation
            self.send_models()

            if i % self.eval_gap == 0:
                evaluated_this_round = True
                print(f"\n-------------Round number: {i+1}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                self.Budget.append(time.time() - s_t)
                best_dice = max(self.rs_test_dice)
                best_miou = max(self.rs_test_miou)

                avg_time = sum(self.Budget) / len(self.Budget) if self.Budget else 0
                remaining = avg_time * (self.global_rounds - i - 1)

                round_bar.set_postfix(
                    Dice=f"{self.rs_test_dice[-1]:.4f}",
                    mIoU=f"{self.rs_test_miou[-1]:.4f}",
                    F1=f"{self.rs_test_f1_dam[-1]:.4f}",
                    BestDice=f"{best_dice:.4f}",
                    BestmIoU=f"{best_miou:.4f}",
                    ETA=f"{remaining/3600:.1f}h",
                )

            

            # -------- per-round CSV row --------
            round_row = {
                "round": i+1,
                "held_out_idx": getattr(self, "held_out_idx", ""),
                "evaluated": int(evaluated_this_round),
                "round_time_sec": self.Budget[-1],
            }
            if evaluated_this_round and self._last_eval_metrics:
                round_row.update(self._last_eval_metrics)
            self.round_logger.log(round_row)

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_miou],
                top_cnt=self.top_cnt
            ):
                break

        print("\nBest segmentation metrics.")
        print(f"Best Pixel Accuracy: {max(self.rs_test_pixel_acc):.4f}")
        print(f"Best Dice: {max(self.rs_test_dice):.4f}")
        print(f"Best mIoU: {max(self.rs_test_miou):.4f}")
        print(f"Best F1-dam: {max(self.rs_test_f1_dam):.4f}")
        if self.Budget:
            avg_round_time = sum(self.Budget) / len(self.Budget)
        else:
            avg_round_time = 0.0

        print("\nAverage time cost per round.")
        print(avg_round_time)

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientSegmentation)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def test_metrics(self):

        # -------- LODO evaluation --------
        if self.protocol == "lodo":
            print(f"SERVER: evaluating held-out client {self.held_out_client.id}")

            confusion, pixels = self.held_out_client.test_metrics()

            return (
                [self.held_out_client.id],
                [pixels],
                [confusion],
            )

        # -------- Standard FL evaluation --------
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        ids = []
        num_pixels = []
        confusions = []

        for c in self.clients:
            confusion, pixels = c.test_metrics()

            ids.append(c.id)
            num_pixels.append(pixels)
            confusions.append(confusion)

        return ids, num_pixels, confusions

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()

        global_confusion = self._sum_confusions(stats[2])
        metrics = segmentation_metrics(global_confusion)
        print("\nConfusion Matrix:")
        print(global_confusion.cpu().numpy())

        print("\nPer-class F1:")
        print(metrics["f1_per_class"])

        print("F1-dam:", metrics["f1_dam"])


        pixel_acc = metrics["pixel_accuracy"]
        dice = metrics["dice"]
        dice_per_class = metrics["dice_per_class"]
        iou = metrics["iou"]
        miou = metrics["mean_iou"]
        f1_dam = metrics["f1_dam"]

        train_loss = 0.0
        client_pixel_accs = []
        client_mious = []
        for confusion in stats[2]:
            client_metrics = segmentation_metrics(confusion)
            client_pixel_accs.append(client_metrics["pixel_accuracy"])
            client_mious.append(client_metrics["mean_iou"])

        if acc == None:
            self.rs_test_acc.append(pixel_acc)
            self.rs_test_pixel_acc.append(pixel_acc)
            self.rs_test_dice.append(dice)
            self.rs_test_dice_per_class.append(dice_per_class)
            self.rs_test_iou.append(iou)
            self.rs_test_miou.append(miou)
            self.rs_test_f1_dam.append(f1_dam)
        else:
            acc.append(pixel_acc)

        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # stash this round's metrics so train() can write them to the CSV
        try:
            iou_mean = float(np.mean(iou))
        except TypeError:
            iou_mean = float(iou)

        self._last_eval_metrics = {
            "pixel_acc": pixel_acc,
            "dice": dice,
            "miou": miou,
            "f1_dam": f1_dam,
            "iou_mean": iou_mean,
            "std_pixel_acc": float(np.std(client_pixel_accs)),
            "std_miou": float(np.std(client_mious)),
            "train_loss": train_loss,
        }

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Pixel Accuracy: {:.4f}".format(pixel_acc))
        print("Averaged Dice: {:.4f}".format(dice))
        print("Averaged F1-dam: {:.4f}".format(f1_dam))
        print("Averaged Mean IoU: {:.4f}".format(miou))
        print("Std Pixel Accuracy: {:.4f}".format(np.std(client_pixel_accs)))
        print("Std Mean IoU: {:.4f}".format(np.std(client_mious)))

    def save_results(self):
        import shutil
        result_path = self._get_result_path()
        run_name = self._get_run_name()

        shutil.copy(
            self.args.config,
            os.path.join(result_path, "config.yaml")
        )
        # --------------------------------------------------
        # Result directory
        # --------------------------------------------------
        

        config_path = os.path.join(result_path, f"{run_name}_config.yaml")

        config = {}

        for k, v in vars(self.args).items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                config[k] = v
            else:
                config[k] = str(v)

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        # --------------------------------------------------
        # Save metrics
        # --------------------------------------------------
        if len(self.rs_test_pixel_acc):

            
            # --------------------------------------------------
            # Append a row to the run-wide summary CSV
            # --------------------------------------------------
            summary_path = os.path.join(result_path, "summary.csv")
            summary_logger = CSVLogger(
                summary_path,
                fieldnames=[
                    "run_name", "run_id", "timestamp", "dataset", "algorithm",
                    "protocol", "held_out_idx", "num_clients", "num_classes",
                    "global_rounds_configured", "global_rounds_completed",
                    "best_pixel_acc", "best_dice", "best_miou", "best_f1_dam",
                    "avg_round_time_sec", "total_time_sec",
                ],
            )
            if len(self.Budget) > 1:
                avg_round_time = sum(self.Budget[1:]) / len(self.Budget[1:])
            elif len(self.Budget) == 1:
                avg_round_time = self.Budget[0]
            else:
                avg_round_time = 0.0

            summary_logger.log({
                "run_name": run_name,
                "run_id": self.run_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset": self.dataset,
                "algorithm": self.algorithm,
                "protocol": getattr(self, "protocol", ""),
                "held_out_idx": getattr(self, "held_out_idx", ""),
                "num_clients": self.num_clients,
                "num_classes": getattr(self, "num_classes", ""),
                "global_rounds_configured": self.global_rounds,
                "global_rounds_completed": len(self.Budget),
                "best_pixel_acc": max(self.rs_test_pixel_acc),
                "best_dice": max(self.rs_test_dice),
                "best_miou": max(self.rs_test_miou),
                "best_f1_dam": max(self.rs_test_f1_dam),
                "avg_round_time_sec": avg_round_time,
                "total_time_sec": sum(self.Budget),
            })
            print("Summary appended to:", summary_path)

    def test_metrics_new_clients(self):
        ids = []
        num_pixels = []
        confusions = []
        for c in self.new_clients:
            confusion, pixels = c.test_metrics()
            ids.append(c.id)
            num_pixels.append(pixels)
            confusions.append(confusion)

        return ids, num_pixels, confusions

    def _sum_confusions(self, confusions):
        if len(confusions) == 0:
            return torch.zeros(self.num_classes, self.num_classes, dtype=torch.int64)

        total = torch.zeros_like(confusions[0])
        for confusion in confusions:
            total += confusion
        return total