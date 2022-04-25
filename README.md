# IFT6759 Deep learning:  ANimal Deep learning Identification ANDI
This repo is for the project of animal classification using deep learning. 

# Team Members
- Jonathan Beaulieu-Emond jonathan.beaulieu-emond@umontreal.ca
- Selim Gilon selim.gilon@umontreal.ca
- Yassine Kassis yassine.kassis.1@umontreal.ca
- Jean-francois Girard-Baril jean-francois.girard-baril@umontreal.ca

Project Gantt chart: https://docs.google.com/spreadsheets/d/1dAW6vDA6k7e2ML3V-6WNNC8MwGt3SMAw1tNjU0YOjMg/edit?usp=sharing

# Project Description

Using the Caltech Camera trap dataset
https://lila.science/datasets/caltech-camera-traps

we trained various models (VGG19, AlexNet, ResNext50, EfficientDet2, Yolo v5)
to perform either classification or detection.

# How to train?

Enter you python environment. If you have a venv :
```
source venv/bin/activate
```
Install the prerequisite
```
python -m pip install -r requirements.txt
sudo apt install zip
```
Download the data :

```
python setup.py
```
then run :
(CAREFUL!!! you cann only go UP in the dataset version, as it will modify the current dataset.
You will be force to download the training set again if you want to go back.)
```
python train.py --dataset {choice of dataset [2,3,4] --model {model name}]}
```

you can call 
```
python train.py -h
```
for more explanation on the parameter of this function.


# How to test your model?


