# PepZOO
📋Directed Evolutionary of Peptides using Multi-Objective Zeroth-Order Optimization

## 📘 Abstract
&nbsp;&nbsp;&nbsp;&nbsp;Antimicrobial peptides (AMPs) emerge as a type of promising therapeutic compounds that exhibit broad spectrum antimicrobial activity with high specificity and good tolerability. Natural AMPs usually need further rational design for improving antimicrobial activity and decreasing toxicity to human cells. Although several algorithms have been developed to optimize AMPs with desired properties, they explored the variations of AMPs in a discrete amino acid sequence space, usually suffering from low efficiency, lack diversity and local optimum. In this work, we propose a novel directed evolution method, named PepZOO, for optimizing multi-properties of AMPs in a continuous representation space guided by multi-objective zeroth-order optimization. PepZOO projects AMPs from a discrete amino acid sequence space into continuous latent representation space by a variational autoencoder. Subsequently, the latent embeddings of prototype AMPs are taken as start points and iteratively updated according to the guidance of multi-objective zeroth-order optimization. Experimental results demonstrate PepZOO outperforms state-of-the-art methods on improving the multi-properties in terms of antimicrobial function, activity, toxicity, and binding affinity to the targets. PepZOO provides a novel research paradigm that optimizes AMPs by exploring property change instead of exploring sequence mutations, accelerating the discovery of potential therapeutic peptides.

## 🧬 Model Structure
<div align=center><img src=img/framework.png></div>

## Required software
CAMP is needed to predict the peptide-protein interaction between generated peptides and the target protein, you should download this tool from github (https://github.com/twopin/CAMP) and put it in PepZOO/tools/. In order to run CAMP correctly, you should download IUPred2A and SCRATCH-1D to get the intrinsic disorder and secondary structure of proteins, please refer to CAMP for usage details. Toxinpred3 is required to predict the toxicity of peptides, you can download it from github (https://github.com/raghavagps/toxinpred3) and use it according to the usage details. All tools should be located in PepZOO/tools/.

## 🚀 How to run?
```
# 1. Clone this repository
git clone https://github.com/chen-bioinfo/PepZOO.git
cd PepZOO

# 2. Creating a virtual environment
conda create -n pepzoo python==3.8
conda activate pepzoo

# 3. the key elements of 'pepzoo' operating environment are listed below(python==3.8):
tensorflo=~=2.2.1
tensorflow-probability==0.10.0
Keras==2.3.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.2
cloudpickle==1.4.1
numpy==1.18.5
pandas==1.1.4
scikit-learn==0.23.2
modlamp==4.2.3
matplotlib==3.3.2
protobuf==3.14.0
seaborn==0.11.0
setuptools==50.3.1
joblib==0.17.0
argparse
tqdm==4.51.0

# 4. generate sequence
conda activate pepzoo
cd PepZOO/run
python run_amp.py 
python run_mic.py
python run_amp_mic.py
python run_cvae.py
python run_pepcvae.py
python run_affinity.py
python run_toxicity.py
python run_affinity_toxicity.py
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
