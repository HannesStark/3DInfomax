# Self-Supervised Graph Representation Learning

**disclaimer: this project is under development and not cleaned up yet**

Contains contrastive learning methods, different contrastive losses, local-global methods but also stuff like "Bootstrap your own Latent" or "Barlow Twins".

Everything is tested on Molecules and with a focus on learning representations that capture 3D information of the molecules 

## Getting started

If you don't have a conda environment, we highly suggest to create one. 

First, clone the repository, then move to the repository and install the dependencies. We highly recommend installing a new environment using conda.

Clone the current repo

    git clone https://github.com/HannesStark/self-supervised-graphs
    cd self-supervised-graphs

Create a new environment with all required packages using `environment.yml` (this can take a while)

    conda env create

Activate the environment and start training with a specific configuration file

    conda activate ssg
    python train.py --config=configs/contrastive_training.yml


