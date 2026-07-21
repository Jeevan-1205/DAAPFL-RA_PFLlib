from flcore.servers.serversegmentation import ServerSegmentation
from flcore.clients.clientfedala import clientFedALA

class FedALA(ServerSegmentation):
    """
    FedALA Server specifically tailored for semantic segmentation.
    It inherits the evaluation and tracking logic from ServerSegmentation
    but ensures that clientFedALA is instantiated instead of the default client.
    """
    def __init__(self, args, times):
        # We must carefully initialize the parent without creating the wrong clients
        # ServerSegmentation.__init__ calls self.set_clients(clientSegmentation)
        # We will let it do that, and then immediately overwrite it.
        super().__init__(args, times)
        
        # Now replace the clients with FedALA clients
        self.clients = []
        self.set_clients(clientFedALA)

    def set_new_clients(self, clientObj=clientFedALA):
        # Ensure new clients are also instantiated as FedALA clients
        super().set_new_clients(clientObj)