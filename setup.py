
import shutil
import subprocess
import sys

#1. Download required files
import gdown
gdown.download("https://drive.google.com/uc?export=download&id=1tKAhkas5sCnIYurURylS63ZgBTdsrg_Q","data/data_split2.zip")
gdown.download("https://drive.google.com/uc?export=download&id=1YXlG77XbMA0a4d12qcUSLa9sMp4rk6Nx","data/unseen_data_test.zip")


#
# wget https://drive.google.com/uc?export=download&id=1tKAhkas5sCnIYurURylS63ZgBTdsrg_Q -O data_split2.zip --show-progress
# wget https://drive.google.com/uc?export=download&id=1YXlG77XbMA0a4d12qcUSLa9sMp4rk6Nx -O unseen_test_set.zip --show-progress
# unzip data_split2.zip
# unzip unseen_test_set.zip
# cd ..


