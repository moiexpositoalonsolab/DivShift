![DivShift logo](https://github.com/moiexpositoalonsolab/DivShift/blob/main/DivShift_logo.png)
# DivShift: Exploring Domain-Specific Distribution Shifts in Large-Scale, Volunteer-Collected Biodiversity Datasets
https://doi.org/10.1609/aaai.v39i27.35060

# [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/35060) | [Extended Version](https://arxiv.org/abs/2410.19816) | [Dataset](https://huggingface.co/datasets/elenagsierra/DivShift-NAWC)

## Overview

The code in this repository implements the data preparation, model training, and evaluation protocols as described in the [DivShift paper](https://ojs.aaai.org/index.php/AAAI/article/view/35060). Using this repository, you can train deep learning models across partitions of five types of bias (spatial, temporal, taxonomic, observer, and sociopolitical) on the [DivShift North American West Coast](https://huggingface.co/datasets/elenagsierra/DivShift-NAWC) (DivShift-NAWC) dataset and test these biases' impact on model performance.

## Data Preparation

### DivShift-NAWC Dataset

Download the DivShift-NAWC dataset from Hugging Face:  
[https://huggingface.co/datasets/elenagsierra/DivShift-NAWC](https://huggingface.co/datasets/elenagsierra/DivShift-NAWC) .

## Run Code

### Requirements

- **Python:** 3.7 or later  
- **Dependencies:**  
  - numpy
  - pandas
  - torch (PyTorch)
  - torchvision
  - scikit-learn
  - requests

> We recommend using a virtual environment to manage dependencies

## Data Preparation

### DivShift-NAWC Dataset

Download the DivShift-NAWC dataset from Hugging Face:  
[https://huggingface.co/datasets/elenagsierra/DivShift-NAWC](https://huggingface.co/datasets/elenagsierra/DivShift-NAWC) .

### Model training

Model training is straightforward and the code provides easy access to choose your train and test bias partitions easily. An example for the observer bias partition would be:
```
python3 src/supervised_train.py --device 0 --data_dir [your_data_directory] --num_epochs 10 --exp_id observer --processes 6 --train_partition spatial_engaged --test_partitions spatial_engaged spatial_casual --model_dir [your_model_directory]
```

### Model testing

Model testing across partitions is as easily straightforward

```
python3 src/supervised_test.py --device 0 --data_dir [your_data_directory] --exp_id engaged_test --processes 6 --train_partition spatial_engaged --test_partition spatial_engaged spatial_casual inat2021mini --model_dir [your_model_directory]
```
## Citation

```
@article{Sierra_Gillespie_Soltani_Exposito-Alonso_Kattenborn_2025,
  author = {Sierra, Elena and Gillespie, Lauren E. and Soltani, Salim and Exposito-Alonso, Moises and Kattenborn, Teja},
  doi = {10.1609/aaai.v39i27.35060},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  month = {Apr.},
  number = {27},
  pages = {28386-28396},
  title = {DivShift: Exploring Domain-Specific Distribution Shifts in Large-Scale, Volunteer-Collected Biodiversity Datasets},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/35060},
  volume = {39},
  year = {2025},
  bdsk-url-1 = {https://ojs.aaai.org/index.php/AAAI/article/view/35060},
  bdsk-url-2 = {https://doi.org/10.1609/aaai.v39i27.35060}
}
```


