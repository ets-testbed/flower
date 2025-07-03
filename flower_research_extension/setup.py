from setuptools import setup, find_packages

setup(
    name="flower_research_extension",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flwr[simulation]>=1.5.0",  # includes simulation dependencies
        "wandb",
        "scikit-learn",
    ],
)

# torch linux: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
