# PepZOO
📋Directed Evolutionary of Peptides using Multi-Objective Zeroth-Order Optimization

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp;Antimicrobial peptides (AMPs) emerge as a type of promising therapeutic compounds that exhibit broad spectrum antimicrobial activity with high specificity and good tolerability. Natural AMPs usually need further rational design for improving antimicrobial activity and decreasing toxicity to human cells. Although several algorithms have been developed to optimize AMPs with desired properties, they explored the variations of AMPs in a discrete amino acid sequence space, usually suffering from low efficiency, lack diversity and local optimum. In this work, we propose a novel directed evolution method, named PepZOO, for optimizing multi-properties of AMPs in a continuous representation space guided by multi-objective zeroth-order optimization. PepZOO projects AMPs from a discrete amino acid sequence space into continuous latent representation space by a variational autoencoder. Subsequently, the latent embeddings of prototype AMPs are taken as start points and iteratively updated according to the guidance of multi-objective zeroth-order optimization. Experimental results demonstrate PepZOO outperforms state-of-the-art methods on improving the multi-properties in terms of antimicrobial function, activity, toxicity, and binding affinity to the targets. PepZOO provides a novel research paradigm that optimizes AMPs by exploring property change instead of exploring sequence mutations, accelerating the discovery of potential therapeutic peptides.

## 🧬 Model Structure
<div align=center><img src=img/framework1.png></div>

## Required software
CAMP is needed to predict the peptide-protein interaction of generated peptides to the target protein, you should download this tool from github (https://github.com/twopin/CAMP) and put it in PepZOO/tools/. In order to run CAMP correctly, you should download IUPred2A and SCRATCH-1D to get the secondary structure and intrinsic disorder of proteins. Please refer to CAMP for usage details.

## 🚀 How to run?
```
# 1. Clone this repository
git clone https://github.com/chen-bioinfo/PepZOO.git
cd PepZOO

# 2. Creating a virtual environment
conda create -n pepzoo python==3.8
conda activate pepzoo

# 3. the key elements of 'pepzoo' operating environment are listed below(python==3.8):
transformers==4.28.1
torch==1.9.0+cu111 (You can download it from the pytorch(https://pytorch.org/get-started/previous-versions) )
peft==0.3.0
modlamp==4.3.0
pandas==1.4.0
datasets==2.12.0
numpy==1.23.5
tqdm==4.65.0
matplotlib==3.7.1
seaborn==0.12.2
tensorboard==2.13.0

# 4. generate sequence
conda activate pepzoo
cd PepZOO/run
python run_amp.py 
python run_mic.py
python run_amp_mic.py
python run_binding.py
python run_toxicity.py
python run_binding_toxicity.py
python run_basic.py
python run_pepcvae.py
```

## 🧐 Analysis
The analysis codes are all located in the folder 'PepZOO/analyze'

| File name | Description |
| ----------- | ----------- |
| case1.ipynb     | Analysis of generated peptides for case1       |
| case2.ipynb     | Analysis of generated peptides for case2       |
| case3.ipynb     | Analysis of generated peptides for case3       |
| PhysicochemicalProperties.ipynb  | Analysis of physicochemicalProperties of generated peptides  |
| affinity.ipynb   | Analysis of generated peptides according to docking energy   |
| affinity_toxicity.ipynb  | Analysis of generated peptides in terms of binding affinity and toxicity  |


## ✏️ Citation
If you use this code or our model for your publication, please cite the original paper:
