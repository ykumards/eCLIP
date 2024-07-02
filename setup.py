from setuptools import setup, find_packages

setup(
    name="eCLIP",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        # other dependencies...
    ],
    author="Yogesh Kumar",
    author_email="ykumards@gmail.com",
    description="Improving Contrastive Learning with Expert Annotations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache",
    url="https://github.com/ykumards/eCLIP",
)
