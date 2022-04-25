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
wget https://drive.google.com/file/d/1QwctLmiog76GoVdyNfo2BMqfFm9gFvqQ/view?usp=sharing
wget https://drive.google.com/file/d/1YXlG77XbMA0a4d12qcUSLa9sMp4rk6Nx/view?usp=sharing
unzip data_split2.zip
unzip unseen_test_set.zip
```
then run :
```
python train.py --dataset {choice of dataset [2,3,4] --model {model name}]}
```

you can call 
```
python train.py -h
```
for more explanation on the parameter of this function.


#How to test your model?


