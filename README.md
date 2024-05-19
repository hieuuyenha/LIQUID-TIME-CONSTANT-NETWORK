# LIQUID-TIME-CONSTANT-NETWORK
LIQUID TIME-CONSTANT NETWORK
# Code run in environment : python==3.7.3
Requisites

numpy==1.18.5

pandas==1.0.5

scipy==1.5.0

tensorflow==1.14.0

scikit-image==0.17.2

scikit-learn==0.23.2

matplotlib==3.2.0

ipykernel==5.1.0

opencv-python==3.4.2.17

opencv-contrib-python==3.4.2.17


# Dataset

With Data for lungCancer.py
link kaggle:https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Another data, we can download all datasets by running
```bash
source download_datasets.sh
```
This script creates a folder ```data```, where all downloaded datasets are stored.
## Training and evaluating the models 


There is exactly one Python module per dataset:
- Room occupancy detection: ```occupancy.py```
- Human activity recognition: ```har.py```
- Lung Pneumoniarecognition
- 
Each script accepts the following four arguments:
- ```--model: lstm | ctrnn | ltc | ltc_rk | ltc_ex```
- ```--epochs: number of training epochs (default 200)```
- ```--size: number of hidden RNN units  (default 32)```
- ```--log: interval of how often to evaluate validation metric (default 1)```
