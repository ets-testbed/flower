from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch.nn as nn

from flower_research_extension.model import Net


def _build_net(num_classes: int) -> nn.Module:
    return Net(num_classes=num_classes)


def _replace_classification_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Patch common classifier heads to match `num_classes`."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        for idx in reversed(range(len(model.classifier))):
            layer = model.classifier[idx]
            if isinstance(layer, nn.Linear):
                model.classifier[idx] = nn.Linear(layer.in_features, num_classes)
                return model
            if isinstance(layer, nn.Conv2d):
                model.classifier[idx] = nn.Conv2d(
                    in_channels=layer.in_channels,
                    out_channels=num_classes,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    dilation=layer.dilation,
                    groups=layer.groups,
                    bias=(layer.bias is not None),
                )
                return model

    if hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
        return model

    if hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
        for idx in reversed(range(len(model.heads))):
            layer = model.heads[idx]
            if isinstance(layer, nn.Linear):
                model.heads[idx] = nn.Linear(layer.in_features, num_classes)
                return model

    raise ValueError(f"Unable to patch classifier head for model type: {type(model).__name__}")


def _build_torchvision_model(model_name: str, num_classes: int) -> nn.Module:
    from torchvision import models as tvm

    if not hasattr(tvm, model_name):
        raise ValueError(f"Model '{model_name}' is not available in this torchvision version")
    constructor = getattr(tvm, model_name)

    try:
        return constructor(weights=None, num_classes=num_classes)
    except TypeError:
        try:
            return constructor(pretrained=False, num_classes=num_classes)
        except TypeError:
            try:
                model = constructor(weights=None)
            except TypeError:
                model = constructor(pretrained=False)
            return _replace_classification_head(model, num_classes)


def _make_torchvision_builder(model_name: str) -> Callable[[int], nn.Module]:
    def _builder(num_classes: int) -> nn.Module:
        return _build_torchvision_model(model_name=model_name, num_classes=num_classes)

    return _builder


TORCHVISION_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext50_32x4d",
    "wide_resnet50_2",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "squeezenet1_1",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
]

MODEL_BUILDERS: Dict[str, Callable[[int], nn.Module]] = {
    "net": _build_net,
    **{model_name: _make_torchvision_builder(model_name) for model_name in TORCHVISION_MODELS},
}

# Informational metadata: expected resource cost by model family.
MODEL_FIT_PROFILE: Dict[str, str] = {
    "net": "light",
    "squeezenet1_1": "light",
    "shufflenet_v2_x1_0": "light",
    "mobilenet_v2": "light",
    "resnet18": "medium",
    "resnet34": "medium",
    "densenet121": "medium",
    "efficientnet_b0": "medium",
    "resnet50": "heavy",
    "resnext50_32x4d": "heavy",
    "wide_resnet50_2": "heavy",
    "convnext_tiny": "heavy",
}

DISTRIBUTION_DESCRIPTIONS: Dict[str, str] = {
    "iid": "Even random split across clients.",
    "dirichlet": "Class proportions per client sampled from Dirichlet(alpha).",
    "inner_dirichlet": "Dirichlet class assignment with client-size weighting.",
    "distribution": "User-provided class-probability matrix per client.",
    "label_skew": "Each client sees only a subset of classes.",
    "pathological": "Alias of label_skew for classic pathological non-IID setups.",
    "shard": "Label-sorted shards assigned to clients.",
    "linear": "Client sizes increase linearly by client index.",
    "square": "Client sizes increase quadratically by client index.",
    "exponential": "Client sizes increase exponentially by client index.",
    "size": "Client sizes follow user-provided partition weights.",
}
DISTRIBUTIONS = list(DISTRIBUTION_DESCRIPTIONS.keys())


@dataclass(frozen=True)
class DatasetModelPolicy:
    default_model: str
    allowed_models: tuple[str, ...]


DATASET_MODEL_POLICIES: Dict[str, DatasetModelPolicy] = {
    "mnist": DatasetModelPolicy(
        default_model="net",
        allowed_models=(
            "net",
            "resnet18",
            "resnet34",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
        ),
    ),
    "fashionmnist": DatasetModelPolicy(
        default_model="net",
        allowed_models=(
            "net",
            "resnet18",
            "resnet34",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
        ),
    ),
    "emnist_balanced": DatasetModelPolicy(
        default_model="resnet18",
        allowed_models=(
            "net",
            "resnet18",
            "resnet34",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
            "densenet121",
            "efficientnet_b0",
        ),
    ),
    "cifar10": DatasetModelPolicy(
        default_model="resnet18",
        allowed_models=(
            "net",
            "resnet18",
            "resnet34",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
            "densenet121",
            "efficientnet_b0",
            "resnet50",
            "resnext50_32x4d",
            "wide_resnet50_2",
            "convnext_tiny",
        ),
    ),
    "svhn": DatasetModelPolicy(
        default_model="resnet18",
        allowed_models=(
            "net",
            "resnet18",
            "resnet34",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
            "densenet121",
            "efficientnet_b0",
            "resnet50",
            "resnext50_32x4d",
            "wide_resnet50_2",
            "convnext_tiny",
        ),
    ),
    "cifar100": DatasetModelPolicy(
        default_model="densenet121",
        allowed_models=(
            "resnet18",
            "resnet34",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "squeezenet1_1",
            "densenet121",
            "efficientnet_b0",
            "resnet50",
            "resnext50_32x4d",
            "wide_resnet50_2",
            "convnext_tiny",
        ),
    ),
}


def resolve_model_name(dataset: str, requested_model: str) -> str:
    policy = DATASET_MODEL_POLICIES.get(dataset)
    if policy is None:
        raise ValueError(f"No dataset-model policy configured for dataset '{dataset}'")
    resolved = policy.default_model if requested_model == "auto" else requested_model
    if resolved not in policy.allowed_models:
        allowed = ", ".join(policy.allowed_models)
        raise ValueError(
            f"Model '{resolved}' is not configured for dataset '{dataset}'. Allowed models: [{allowed}]"
        )
    return resolved


def build_capabilities(datasets: list[str]) -> dict:
    return {
        "datasets": datasets,
        "distributions": DISTRIBUTIONS,
        "distribution_descriptions": DISTRIBUTION_DESCRIPTIONS,
        "models": sorted(MODEL_BUILDERS.keys()),
        "model_fit_profile": {
            model: MODEL_FIT_PROFILE.get(model, "medium") for model in sorted(MODEL_BUILDERS.keys())
        },
        "dataset_model_policies": {
            dataset: {
                "default_model": policy.default_model,
                "allowed_models": list(policy.allowed_models),
            }
            for dataset, policy in sorted(DATASET_MODEL_POLICIES.items())
        },
    }
