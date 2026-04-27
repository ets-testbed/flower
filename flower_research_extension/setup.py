from setuptools import setup, find_packages

setup(
    name="flower_research_extension",
    version="0.2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "flwr[simulation]>=1.5.0",
        "numpy>=1.24",
        "PyYAML>=6.0",
        "torch>=2.2",
        "torchvision>=0.17",
        "wandb>=0.16",
        "scikit-learn>=1.3",
    ],
    extras_require={
        "dev": ["pytest>=8.0"],
    },
)
