# SynergyX: a Multi-Modality Mutual Attention Network for interpretable drug synergy prediction

## Introduction
Drug combination therapy has shown promising results in enhancing treatment efficacy and overcoming drug resistance in clinic. However, the complex interactions between drugs and cancer cells pose significant challenges in accurately predicting synergy outcomes. We introduce SynergyX, a Multi-Modality Mutual Attention Network that effectively integrates multi-omics data and captures multi-dimensional cross-modal interactions, resulting in improved drug synergy prediction. 


## Overview
The repository is organised as follows:
- `data/` contains data files and data processing files;
- `dataset/` contains the necessary files for creating the dataset;
- `models/` contains different modules of SynergyX;
- `saved_model/` contains the trained weights of SynergyX;
- `experiment/` contains log files and output files;
- `utils.py` contains the necessary processing subroutines;
- `metrics.py` contains the necessary functions to calculate model evaluation metrics;
- `main.py` main function for SynergyX.


## Requirements
The SynergyX network is built using PyTorch and PyTorch Geometric. You can use following commands to create conda env with related dependencies.

```
conda create -n synergyx python=3.8 pip=22.1.2
conda activate synergyx
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
pip install torch-geometric==2.2
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html
pip install pandas
pip install subword-nmt
pip install rdkit
```

## Implementation
### Data Preprocessing

1. In SynergyX, we use normalized multi-omics data of cell lines and substructure encodings of drugs as input features. The preprocessed data is stored in `data/0_cell_data/` and `data/1_drug_data/`. You can also preprocess your own data by following the steps in `data/data_process.ipynb`. <br>
**Note**:
   Due to the size of certain data files, direct uploading to GitHub is not feasible.These files can be accessed for download [here](https://drive.google.com/drive/folders/1jhzBSWNth5Clv9DQj8M7XujT1HIAfkoM?usp=drive_link). If you require these data, please download it and place it in the appropriate folder.
   
3. Our collected drug combination data is stored in `data/split/all_items.npy`, where the data items are in the format `[drugA_canonical_smi, drugB_canonical_smi, cell_ID, label]`. Note that your data should be organized in the same format. If the label is unknown, assign a value of `0.` to the label.
4. Before training or testing the model, remember to update the data path within `utils.py`.

### Model Training

Run the following commands to train SynergyX. 

``` 
python main.py --mode train  > './experiment/'$(date +'%Y%m%d_%H%M').log
``` 


### Model Testing/Inferencing

You can use our pre-trained model for drug synergy prediciton. The trained model is available for download [here](https://drive.google.com/file/d/1QuyJw_ISQIv66YmZyZKBKCwm7kENg29q/view?usp=drive_link). Please store it in the `saved_models/`, and proceed to execute the following commands.


1. Run the following commands to test SynergyX. Specify the path of pre-trained model with `saved-model` parameter.
    ```
    python main.py --mode test --save-model ./saved_model/0_fold_SynergyX.pt > './experiment/'$(date +'%Y%m%d_%H%M').log
    ```

2. If data's labels are unknown, we recommend using the `Infer` mode. Please specify your own dataset path using the `infer-path` and set `output-attn` to 1 if you require the attention matrix for further analysis. The prediction results will be stored in the `./experiment` directory.
    ```
    python main.py --mode infer --infer-path ./data/infer_data/sample_infer_items.npy --output-attn 0 > './experiment/'$(date +'%Y%m%d_%H%M').log
    ```
