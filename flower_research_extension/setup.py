from setuptools import setup, find_packages

setup(
    name="flower_research_extension",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "flwr>=1.5.0",  # match your Flower version
        "wandb",
    ],
)
# pip install -e .
