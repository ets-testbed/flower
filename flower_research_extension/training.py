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


from typing import Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def evaluate(server_round: int, parameters, config: Dict, device=DEVICE) -> Tuple[float, Dict]:
    model = Net().to(device)
    set_parameters(model, parameters)
    _, testloader = load_cifar10_partition(0, NUM_PARTITIONS)

    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    batch_accuracies = []

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            batch_accuracy = (preds == labels).sum().item() / len(labels)
            batch_accuracies.append(batch_accuracy)

    loss = total_loss / len(testloader.dataset)
    accuracy = sum([p == t for p, t in zip(y_pred, y_true)]) / len(y_true)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return float(loss), {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "accuracy_min": float(np.min(batch_accuracies)),
        "accuracy_max": float(np.max(batch_accuracies)),
        "accuracy_std": float(np.std(batch_accuracies)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }






