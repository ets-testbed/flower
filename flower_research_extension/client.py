import torch
from flwr.client import NumPyClient
from flwr.common import Context

from flower_research_extension.model import Net, get_parameters, set_parameters
from flower_research_extension.data_files.cifar10 import load_cifar10_partition

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FlowerClient(NumPyClient):
    def __init__(self, cid, model, trainloader, valloader):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self._train(epochs=config.get("local_epochs", 1))
        loss, acc = self._evaluate(self.valloader)
        return get_parameters(self.model), len(self.trainloader.dataset), {"loss": loss, "accuracy": acc}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, acc = self._evaluate(self.valloader)
        return float(loss), len(self.valloader.dataset), {"loss": loss, "accuracy": acc}

    def _train(self, epochs):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch in self.trainloader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def _evaluate(self, loader):
        self.model.eval()
        total, correct, loss_total = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in loader:
                images, labels = batch["img"].to(DEVICE), batch["label"].to(DEVICE)
                outputs = self.model(images)
                loss_total += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return loss_total / len(loader.dataset), correct / total


def client_fn(context: Context):
    cid = context.node_config["partition-id"]
    nparts = context.node_config["num-partitions"]

    model = Net().to(DEVICE)
    trainloader, valloader = load_cifar10_partition(cid, nparts)
    return FlowerClient(cid, model, trainloader, valloader).to_client()
