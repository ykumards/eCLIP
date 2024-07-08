from setuptools import setup, find_packages

setup(
    name="eCLIP",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets==2.15.0",
        "hydra-core==1.3.0",
        "numpy==1.26.2",
        "pandas==2.1.3",
        "peft==0.6.2",
        "plotly==5.22.0",
        "pytorch-lightning==2.1.2",
        "rich==13.7.1",
        "scikit-image==0.24.0",
        "scikit-learn==1.3.2",
        "torch==2.1.1",
        "torchvision==0.16.1",
        "transformers==4.35.2",
        "wandb==0.16.0"
    ],
    author="Yogesh Kumar",
    author_email="ykumards@gmail.com",
    description="Improving Contrastive Learning with Expert Annotations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/ykumards/eCLIP",
)
