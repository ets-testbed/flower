import torch
from typing import Dict, Tuple
from flower_research_extension.model import Net, set_parameters
from flower_research_extension.data_files.cifar10 import load_cifar10_partition


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PARTITIONS = 20


def fit_config(server_round: int) -> Dict:
    return {
        "server_round": server_round,
        "local_epochs": 5,
    }


def evaluate(server_round: int, parameters, config: Dict, device=DEVICE) -> Tuple[float, Dict]:
    model = Net().to(device)
    set_parameters(model, parameters)
    _, valloader = load_cifar10_partition(0, NUM_PARTITIONS)

    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in valloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    loss = total_loss / len(valloader.dataset)
    accuracy = correct / total
    return loss, {"accuracy": accuracy}
