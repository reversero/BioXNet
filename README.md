# BioXNet: a biologically inspired neural network for deciphering anti-cancer drug response in precision medicine
This is the code repository for the paper "BioXNet: a biologically inspired neural network for deciphering anti-cancer drug response in precision medicine".
## Overview
BioXNet is a novel deep learning model for anti-cancer drug response prediction, which captures drug response mechanisms by seamlessly integrating drug target information with genomic profiles (genetic and epigenetic modifications) into a single biologically inspired neural network. BioXNet exhibited superior performance in drug response prediction tasks in both preclinical and clinical settings. The analysis of BioXNet's interpretability revealed its ability to identify significant differences in drug response mechanisms between cell lines and the human body. Notably, the key factor of drug response is the drug targeting genes in cell lines but methylation modifications in the human body. We also developed an [online human-readable interface](https://huggingface.co/spaces/Jayet010/bioxnet) of BioXNet for drug response exploration by medical professionals and laymen. BioXNet represents a step further towards unifying drug, cell line and patients’ data under a holistic interpretable machine learning framework for precision medicine in cancer therapy.
## Software requirements
The entire codebase is written in python. Package requirements are as follows:
- python=3.11.5
- pytorch=2.0.1
- scikit-learn=1.3.0
- numpy=1.24.3
- pandas=2.0.3
- tqdm
## Data requirements
All the data necessary to train the BioXNet model can be found at this [link](https://drive.google.com/file/d/1roNYMD1R3Qhed7f6dkzDNDNwSq7T5QRg/view?usp=sharing). Please download and unzip the data, and then place it in the `data` folder.
## How to train
### Train on the GDSC
For the training of BioXNet on the GDSC dataset, please use the following command
```bash
python bioxnet_train.py --config config/gdsc.json
```
### Train on the TCGA without weight transfer
For the training of BioXNet on the TCGA dataset without weight transfer, please use the following command
```bash
python bioxnet_train.py --config config/tcga_multiclass_nopretrain.json
```
### Train on the TCGA with weight transfer
For the training of BioXNet on the TCGA dataset with weight transfer, please modify the configuration file `config/tcga_multiclass_withpretrain.json` by replacing the path of the trained model on the GDSC dataset (e.g., `saved/models/BioXNet_GDSC/1020_143614/model_best.pth`). Afterward, use the following command
```bash
python bioxnet_train.py --config config/tcga_multiclass_withpretrain.json
```
## License
This project is available under the MIT license.
## Contact
Jiannan Yang - jnyang@hku.hk 
