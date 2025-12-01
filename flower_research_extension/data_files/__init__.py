from .registry import REGISTRY
from .cifar10_provider import CIFAR10Provider
from .mnist_provider import MNISTProvider
from .cifar100_provider import CIFAR100Provider
from .fashionmnist_provider import FashionMNISTProvider
from .svhn_provider import SVHNProvider
from .emnist_provider import EMNISTBalancedProvider

# Register built-ins
REGISTRY.register(CIFAR10Provider())
REGISTRY.register(MNISTProvider())
REGISTRY.register(CIFAR100Provider())
REGISTRY.register(FashionMNISTProvider())
REGISTRY.register(SVHNProvider())
REGISTRY.register(EMNISTBalancedProvider())

__all__ = ["REGISTRY"]
