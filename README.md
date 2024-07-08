# eCLIP
Code for "Improving Medical Multi-modal Contrastive Learning with Expert Annotations" accepted at ECCV 2024.

![image](https://github.com/ykumards/eCLIP/assets/5177126/f1ec7a9a-3a5e-47e1-9150-ca2e474fa4f1)


### Setup

We use the `eclip` package for training and evaluation. To install it run `pip install -e .` from the root folder and it installs all the dependencies. 

We use `pytorch-lightning` to train the model and `hydra` to handle the config files. 


### Training model
CLIP/eCLIP training can be done by running by passing the appropriate hydra flags for config. Toggle the `use_expert` flag to switch between eCLIP and CLIP pretraining.

```
python eclip/train.py data=data_default hydra=hydra_default \
    use_expert=true \
    model="model_default" \
    batch_size=64 \
    scheduler_name="cosine" \
    max_length=256 \
    precision="32" \
    learning_rate=1e-4 \
    weight_decay=1e-3 \
    max_steps=200 \
    val_check_interval=20 \
    limit_val_batches=10 \
    wandb_project_name="eclip-debug" \
    num_gpus=1
```

### Evaluation

The evaluation modules are in `eclip/eval`. For example to report zero-shot performance on Chexpert, run the following after updating the appropriate paths

```
python eclip/eval/eval_classification_chexpert.py
```

It should print the ZS accuracy and F1 scores for the models

![image](https://github.com/ykumards/eCLIP/assets/5177126/34071f87-3d70-43c8-8ac3-548e4275ae28)


#### Report Generation

We use `Mistral 7B Instruction` for generating radiology reports using the input image. This utilizes Retrieval Augmented Generation (RAG) and uses nearest neighbors to pick some relevant reports and injects them in the LLM prompt.

![image](https://github.com/ykumards/eCLIP/assets/5177126/60be85ac-dbe4-4321-b7e9-8dcda9231202)
