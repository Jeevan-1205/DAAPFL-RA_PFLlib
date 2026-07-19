import os
import time
import copy
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
        self.rs_test_f1_per_class = []

        # CSV logging
        self.run_id = make_run_id()
        self.round_logger = None
        self._last_eval_metrics = {}

        if self.algorithm == "SCAFFOLD":
            self.server_learning_rate = getattr(args, "server_learning_rate", 1.0)
            self.global_c = []
            for param in self.global_model.parameters():
                self.global_c.append(torch.zeros_like(param))
        elif self.algorithm == "pFedMe":
            self.beta = getattr(args, "beta", 0.0)

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

        fieldnames = [
            "round", "held_out_idx", "evaluated",
            "pixel_acc", "dice", "miou", "iou_mean","f1_dam",
            "std_pixel_acc", "std_miou", "train_loss",
            "round_time_sec",
        ]
        classes = ["bg", "no_damage", "minor", "major", "destroyed"]
        for p in classes:
            fieldnames.append(f"dice_{p}")
        for p in classes:
            fieldnames.append(f"iou_{p}")
        for p in classes:
            fieldnames.append(f"f1_{p}")

        self.round_logger = CSVLogger(
            round_csv_path,
            fieldnames=fieldnames,
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

        if self.algorithm != "Local":
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
            if self.algorithm != "Local":
                self.send_models()

            for client in self.selected_clients:
                client.train()

            if self.algorithm != "Local":
                if self.algorithm == "pFedMe":
                    self.previous_global_model = copy.deepcopy(list(self.global_model.parameters()))
                self.receive_models()

            if self.dlg_eval and i % self.dlg_gap == 0 and self.algorithm != "Local":
                self.call_dlg(i)

            self.log_federated_update_statistics(i)
            self.aggregate_parameters()
            if self.algorithm == "pFedMe":
                self.beta_aggregate_parameters()

            # Broadcast aggregated model before evaluation
            if self.algorithm != "Local":
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

    def send_models(self):
        if self.algorithm == "SCAFFOLD":
            assert (len(self.clients) > 0)
            for client in self.clients:
                start_time = time.time()
                client.set_parameters(self.global_model, self.global_c)
                client.send_time_cost['num_rounds'] += 1
                client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
        else:
            super().send_models()

    def receive_models(self):
        if self.algorithm == "Local":
            return
        super().receive_models()
        if self.algorithm in ("FedPer", "FedRep"):
            for j, l_model in enumerate(self.uploaded_models):
                if hasattr(l_model, "encoder"):
                    self.uploaded_models[j] = copy.deepcopy(l_model.encoder)
                elif hasattr(l_model, "base"):
                    self.uploaded_models[j] = copy.deepcopy(l_model.base)
                else:
                    self.uploaded_models[j] = copy.deepcopy(l_model)
        else:
            for j, l_model in enumerate(self.uploaded_models):
                self.uploaded_models[j] = copy.deepcopy(l_model)

    def aggregate_parameters(self):
        if self.algorithm == "Local":
            return
        
        if self.algorithm == "SCAFFOLD":
            assert len(self.uploaded_ids) > 0
            global_model = copy.deepcopy(self.global_model)
            global_c = copy.deepcopy(self.global_c)
            for cid in self.uploaded_ids:
                dy, dc = self.clients[cid].delta_yc()
                for server_param, client_param in zip(global_model.parameters(), dy):
                    server_param.data += client_param.data.clone() / self.num_join_clients * self.server_learning_rate
                for server_param, client_param in zip(global_c, dc):
                    server_param.data += client_param.data.clone() / self.num_clients
            self.global_model = global_model
            self.global_c = global_c
            return

        if self.algorithm in ("FedPer", "FedRep"):
            assert len(self.uploaded_models) > 0
            if hasattr(self.global_model, "encoder"):
                target_module = self.global_model.encoder
            elif hasattr(self.global_model, "base"):
                target_module = self.global_model.base
            else:
                target_module = self.global_model

            for param in target_module.parameters():
                param.data.zero_()

            for w, client_module in zip(self.uploaded_weights, self.uploaded_models):
                for target_param, source_param in zip(target_module.parameters(), client_module.parameters()):
                    target_param.data += source_param.data.clone() * w
        else:
            super().aggregate_parameters()

    def beta_aggregate_parameters(self):
        # aggregate avergage model with previous model using parameter beta
        for pre_param, param in zip(self.previous_global_model, self.global_model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

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

        pixel_acc = metrics["pixel_accuracy"]
        dice = metrics["dice"]
        dice_per_class = metrics["dice_per_class"]
        iou = metrics["iou"]
        miou = metrics["mean_iou"]
        f1_dam = metrics["f1_dam"]
        f1_per_class = metrics["f1_per_class"]

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
            self.rs_test_f1_per_class.append(f1_per_class)
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

        dice_bg, dice_no_damage, dice_minor, dice_major, dice_destroyed = dice_per_class
        iou_bg, iou_no_damage, iou_minor, iou_major, iou_destroyed = iou
        f1_bg = float('nan')
        f1_no_damage, f1_minor, f1_major, f1_destroyed = f1_per_class

        self._last_eval_metrics = {
            "pixel_acc": pixel_acc,
            "dice": dice,
            "miou": miou,
            "f1_dam": f1_dam,
            "iou_mean": iou_mean,
            "std_pixel_acc": float(np.std(client_pixel_accs)),
            "std_miou": float(np.std(client_mious)),
            "train_loss": train_loss,
            "dice_bg": dice_bg,
            "dice_no_damage": dice_no_damage,
            "dice_minor": dice_minor,
            "dice_major": dice_major,
            "dice_destroyed": dice_destroyed,
            "iou_bg": iou_bg,
            "iou_no_damage": iou_no_damage,
            "iou_minor": iou_minor,
            "iou_major": iou_major,
            "iou_destroyed": iou_destroyed,
            "f1_bg": f1_bg,
            "f1_no_damage": f1_no_damage,
            "f1_minor": f1_minor,
            "f1_major": f1_major,
            "f1_destroyed": f1_destroyed,
        }

        print("\nPer-class Dice")
        print(f"Background : {dice_bg:.4f}")
        print(f"No Damage  : {dice_no_damage:.4f}")
        print(f"Minor      : {dice_minor:.4f}")
        print(f"Major      : {dice_major:.4f}")
        print(f"Destroyed  : {dice_destroyed:.4f}")

        print("\nPer-class IoU")
        print(f"Background : {iou_bg:.4f}")
        print(f"No Damage  : {iou_no_damage:.4f}")
        print(f"Minor      : {iou_minor:.4f}")
        print(f"Major      : {iou_major:.4f}")
        print(f"Destroyed  : {iou_destroyed:.4f}")

        print("\nPer-class F1")
        print("Background : N/A")
        print(f"No Damage  : {f1_no_damage:.4f}")
        print(f"Minor      : {f1_minor:.4f}")
        print(f"Major      : {f1_major:.4f}")
        print(f"Destroyed  : {f1_destroyed:.4f}")

        print("\nConfusion Matrix")
        print(global_confusion.cpu().numpy())

        print("\nAveraged Train Loss: {:.4f}".format(train_loss))
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
            fieldnames = [
                "run_name", "run_id", "timestamp", "dataset", "algorithm",
                "protocol", "held_out_idx", "num_clients", "num_classes",
                "global_rounds_configured", "global_rounds_completed",
                "best_pixel_acc", "best_dice", "best_miou", "best_f1_dam",
            ]
            classes = ["bg", "no_damage", "minor", "major", "destroyed"]
            for p in classes:
                fieldnames.append(f"best_dice_{p}")
            for p in classes:
                fieldnames.append(f"best_iou_{p}")
            for p in classes:
                fieldnames.append(f"best_f1_{p}")
            fieldnames.extend(["avg_round_time_sec", "total_time_sec"])

            summary_logger = CSVLogger(
                summary_path,
                fieldnames=fieldnames,
            )
            if len(self.Budget) > 1:
                avg_round_time = sum(self.Budget[1:]) / len(self.Budget[1:])
            elif len(self.Budget) == 1:
                avg_round_time = self.Budget[0]
            else:
                avg_round_time = 0.0

            best_idx = np.argmax(self.rs_test_miou) if len(self.rs_test_miou) > 0 else -1
            if best_idx >= 0:
                best_dice_per_class = self.rs_test_dice_per_class[best_idx]
                best_iou_per_class = self.rs_test_iou[best_idx]
                best_f1_per_class = self.rs_test_f1_per_class[best_idx]
                best_pixel_acc = max(self.rs_test_pixel_acc)
                best_dice = max(self.rs_test_dice)
                best_miou = max(self.rs_test_miou)
                best_f1_dam = max(self.rs_test_f1_dam)
            else:
                best_dice_per_class = [0.0]*5
                best_iou_per_class = [0.0]*5
                best_f1_per_class = [0.0]*4
                best_pixel_acc, best_dice, best_miou, best_f1_dam = 0.0, 0.0, 0.0, 0.0

            log_dict = {
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
                "best_pixel_acc": best_pixel_acc,
                "best_dice": best_dice,
                "best_miou": best_miou,
                "best_f1_dam": best_f1_dam,
                "avg_round_time_sec": avg_round_time,
                "total_time_sec": sum(self.Budget),
            }

            for i, p in enumerate(classes):
                log_dict[f"best_dice_{p}"] = best_dice_per_class[i]
                log_dict[f"best_iou_{p}"] = best_iou_per_class[i]
                if i == 0:
                    log_dict[f"best_f1_{p}"] = float('nan')
                else:
                    log_dict[f"best_f1_{p}"] = best_f1_per_class[i-1]

            summary_logger.log(log_dict)
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

    def log_federated_update_statistics(self, round_idx):
        if not hasattr(self, "uploaded_models") or not self.uploaded_models or self.algorithm == "Local":
            return

        with torch.no_grad():
            client_deltas = []
            client_layer_norms = []
            layer_names = ["encoder", "fusion", "attention", "decoder", "fc"]

            target_global = self.global_model
            if self.algorithm in ("FedPer", "FedRep"):
                if hasattr(self.global_model, "encoder"):
                    target_global = self.global_model.encoder
                elif hasattr(self.global_model, "base"):
                    target_global = self.global_model.base

            def get_layer_delta(local_m, global_m, layer_name):
                if not hasattr(local_m, layer_name) or not hasattr(global_m, layer_name):
                    return None
                l_mod = getattr(local_m, layer_name)
                g_mod = getattr(global_m, layer_name)
                diffs = []
                for p_loc, p_glob in zip(l_mod.parameters(), g_mod.parameters()):
                    diffs.append((p_loc.data - p_glob.data).view(-1))
                if not diffs:
                    return None
                return torch.cat(diffs)

            def get_full_delta(local_m, global_m):
                diffs = []
                for p_loc, p_glob in zip(local_m.parameters(), global_m.parameters()):
                    diffs.append((p_loc.data - p_glob.data).view(-1))
                return torch.cat(diffs)

            delta_agg = None
            for idx, (w, l_model) in enumerate(zip(self.uploaded_weights, self.uploaded_models)):
                delta = get_full_delta(l_model, target_global)
                client_deltas.append(delta)
                if delta_agg is None:
                    delta_agg = delta.clone() * w
                else:
                    delta_agg += delta * w

                l_norms = {}
                if self.algorithm in ("FedPer", "FedRep"):
                    for l_name in layer_names:
                        if l_name == "encoder":
                            l_norms[l_name] = torch.norm(delta, p=2).item()
                        else:
                            l_norms[l_name] = 0.0
                else:
                    for l_name in layer_names:
                        l_delta = get_layer_delta(l_model, self.global_model, l_name)
                        if l_delta is not None and l_delta.numel() > 0:
                            l_norms[l_name] = torch.norm(l_delta, p=2).item()
                        else:
                            l_norms[l_name] = 0.0
                client_layer_norms.append(l_norms)

            delta_agg_norm = torch.norm(delta_agg, p=2).item() if delta_agg is not None else 0.0
            safe_agg_norm = delta_agg_norm if delta_agg_norm > 0 else 1e-12

            client_norms = [torch.norm(d, p=2).item() for d in client_deltas]
            cos_to_agg = []
            for d, norm_d in zip(client_deltas, client_norms):
                if norm_d == 0:
                    cos_to_agg.append(0.0)
                else:
                    cos = torch.dot(d, delta_agg).item() / (norm_d * safe_agg_norm)
                    cos_to_agg.append(cos)

            num_active = len(self.uploaded_ids)
            pairwise_matrix = np.zeros((num_active, num_active))
            for j in range(num_active):
                pairwise_matrix[j, j] = 1.0
                for k in range(j + 1, num_active):
                    if client_norms[j] == 0 or client_norms[k] == 0:
                        sim = 0.0
                    else:
                        sim = torch.dot(client_deltas[j], client_deltas[k]).item() / (client_norms[j] * client_norms[k])
                    pairwise_matrix[j, k] = sim
                    pairwise_matrix[k, j] = sim

            disaster_names = [
                'Earthquake', 'Flood', 'Hurricane', 'Tornado',
                'Tsunami', 'Volcano', 'Wildfire',
            ]
            print(f"\n====================================================================================")
            print(f"Round {round_idx + 1} — Client Update Statistics (Pre-Aggregation)")
            print(f"====================================================================================")

            print("\n1. Per-Client Update Norms & Cosine Similarity to Aggregated Update:")
            print(f"{'Client ID':<22} | {'Weight (w_i)':<12} | {'L2 Norm ||W_loc - W_glob||':<26} | {'Cos Sim to Aggregated':<22}")
            print("-" * 88)
            for j, cid in enumerate(self.uploaded_ids):
                d_name = disaster_names[cid] if cid < len(disaster_names) else f"ID_{cid}"
                label = f"Client {cid} ({d_name[:12]})"
                print(f"{label:<22} | {self.uploaded_weights[j]:<12.4f} | {client_norms[j]:<26.6f} | {cos_to_agg[j]:<22.4f}")
            print(f"{'Aggregated Update':<22} | {'1.0000':<12} | {delta_agg_norm:<26.6f} | {'1.0000':<22}")

            print("\n2. Pairwise Cosine Similarity Between Client Updates:")
            header_cols = [f"C{cid}" for cid in self.uploaded_ids]
            print(f"{'':<22} | " + " | ".join([f"{col:>7}" for col in header_cols]))
            print("-" * (25 + 10 * num_active))
            for j, cid in enumerate(self.uploaded_ids):
                d_name = disaster_names[cid] if cid < len(disaster_names) else f"ID_{cid}"
                label = f"Client {cid} ({d_name[:12]})"
                row_strs = [f"{pairwise_matrix[j, k]:7.4f}" for k in range(num_active)]
                print(f"{label:<22} | " + " | ".join(row_strs))

            print("\n3. Per-Layer Update Norms (||W_local,layer - W_global,layer||):")
            print(f"{'Client ID':<22} | {'Encoder':<12} | {'Fusion':<12} | {'Attention':<12} | {'Decoder':<12} | {'Seg Head (fc)':<14}")
            print("-" * 94)
            for j, cid in enumerate(self.uploaded_ids):
                d_name = disaster_names[cid] if cid < len(disaster_names) else f"ID_{cid}"
                label = f"Client {cid} ({d_name[:12]})"
                ln = client_layer_norms[j]
                print(f"{label:<22} | {ln['encoder']:<12.4f} | {ln['fusion']:<12.4f} | {ln['attention']:<12.4f} | {ln['decoder']:<12.4f} | {ln['fc']:<14.4f}")
            print(f"====================================================================================\n")