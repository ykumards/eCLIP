# eCLIP
Code for "Improving Medical Multi-modal Contrastive Learning with Expert Annotations"

#### Setup

We use the `eclip` package for training and evaluation. To install it run `pip install -e .` from the root folder and it installs all the dependencies. 

We use `pytorch-lightning` to train the model and `hydra` to handle the config files. The evaluation modules are in `eclip/eval`. 
