"""
DAY 1:
!git clone https://github.com/ultralytics/yolov5  # clone
%cd yolov5
on veut être dans models/yolov5
% unzip le yolo.zip dans ce fichier : models/yolov5/
il faut que le data.yaml soit dans ce fichier ainsi que les fichiers d'images et de labels:
models/yolov5/yolo/train|test|valid/images|labels
%pip install -qr requirements.txt
train: on veut jouer avec le batch size et avoir le maximum qui ne plante pas (au pire tu mets 30-40 et c'est parti)
!python train.py --img 600 --batch 80 --epochs 50 --patience 5 --data data.yaml --weights yolov5l.pt --cache
on peut juste essayer avec une seule epoch et voir qu'il y aura un log de l'experience qui est créé dans
models/yolov5/runs/train/exp/
"""


"""
DAY 2:
Run 320 or 608 depending on the outcome of the 608 (see tonight) on a bigger model (medium? large?). ADD WANDB before
"""

"""
DAY 3:
RESULTS + faster rcnn etc
"""