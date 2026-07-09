import torch
from flcore.clients.clientavg import clientAVG
from utils.segmentation_metrics import segmentation_confusion_matrix


class clientSegmentation(clientAVG):
    """FedAvg client with semantic-segmentation evaluation metrics."""

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.segmentation_ignore_index = getattr(args, "segmentation_ignore_index", None)

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.eval()

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

                output = self.model(x)
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

        return confusion, int(confusion.sum().item())
