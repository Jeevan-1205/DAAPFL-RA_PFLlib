import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            print(f"Epoch {epoch}")

            prev_end = time.time()

            for batch_idx, (x, y) in enumerate(trainloader):

                # Time spent waiting for the next batch
                load_time = time.time() - prev_end

                transfer_start = time.time()

                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)

                y = y.to(self.device)

                transfer_time = time.time() - transfer_start

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                forward_start = time.time()

                output = self.model(x)
                loss = self.loss(output, y)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                forward_time = time.time() - forward_start

                backward_start = time.time()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.device == "cuda":
                    torch.cuda.synchronize()

                backward_time = time.time() - backward_start

                prev_end = time.time()

                if batch_idx % 100 == 0:
                    print(
                        f"Batch {batch_idx} | "
                        f"Load={load_time:.3f}s | "
                        f"Transfer={transfer_time:.3f}s | "
                        f"Forward={forward_time:.3f}s | "
                        f"Backward={backward_time:.3f}s"
                    )

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
