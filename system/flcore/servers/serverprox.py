from flcore.clients.clientprox import clientProx
from flcore.clients.clientsegmentation import clientSegmentation
from flcore.servers.serversegmentation import ServerSegmentation


class clientProxSegmentation(clientProx):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.segmentation_ignore_index = getattr(args, "segmentation_ignore_index", None)

    test_metrics = clientSegmentation.test_metrics


class FedProx(ServerSegmentation):
    def set_clients(self, clientObj=clientProxSegmentation):
        super().set_clients(clientProxSegmentation)

    def set_new_clients(self, clientObj=clientProxSegmentation):
        super().set_new_clients(clientProxSegmentation)
