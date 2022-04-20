# import json
# import numpy as np
# results=json.load(open("best_predictions.json"))
#
# data={}
# a=[]
# for result in results :
#     image_id=result["image_id"]
#     category_id,bbox,score=result["category_id"],result["bbox"],result["score"]
#     temp=data.get(image_id,value=a)
#     data[image_id]=temp.append([category_id,bbox[0],bbox[1],bbox[2],bbox[3],score])
#
#
# stop=1
#
# for key in data.keys() :
#     results=data[key]
import os
import numpy as np
from sklearn.metrics import confusion_matrix
num_classes=15
# ------------defining metrics--------------------------------------------
import sklearn
from sklearn.metrics import top_k_accuracy_score



def macro(true, pred):


    return sklearn.metrics.f1_score(true, pred, average='macro')  # weighted??


def f1(true, pred):

    return sklearn.metrics.f1_score(true, pred, average='weighted')  # weighted??


def precision(true, pred):

    return sklearn.metrics.precision_score(true, pred, average='macro')


def recall(true, pred):
    return sklearn.metrics.recall_score(true, pred, average='macro')


metrics = {
    "f1": f1,
    "macro": macro,
    "precision": precision,
    "recall": recall,
}

folder="labels2/content/yolov5/runs/detect/exp5/labels"

mapping={}
for file in os.listdir("data/data/test_set3/test/labels") :
    file_id=file.split("_")[0]
    mapping[file_id]=file

results=[[],[]]

multicount=False
for file in os.listdir(folder) :

    file_id=file.split("_")[0]
    true_file=mapping.pop(file_id)

    true_label=np.loadtxt("data/data/test_set3/test/labels/"+true_file,unpack=True)
    a=(len(true_label.flatten())+1)//5
    true_label=true_label.reshape(a,5)
    label = np.loadtxt(folder + "/" + file, unpack=True)
    a = (len(label.flatten()) + 1) // 5
    label = label.reshape(a, 5)
    count=0
    if multicount :
        for true in true_label :
            try :
                results[0].append(int(true[0,0]))
            except :
                results[0].append(int(true[ 0]))
            count+=1



        for pred in label :
            try :
                results[1].append(int(pred[0,0]))
            except :
                results[1].append(int(pred[0]))
            count-=1

        while count>0 :
            results[1].append(15) # le modele ne trouvait rien et predisait donc la classe nulle
            count-=1
        while count<0 :#le modele a predit des classes d'extra
            results[0].append(15)
            count+=1

    else :
        results[0].append(true_label.flatten()[0])
        results[1].append(label.flatten()[0])

for key in mapping.keys() :
    file=mapping[key]
    pred=15
    try :
        label = np.loadtxt(folder + "/" + file, unpack=True).flatten()[0]
    except :
        label=15
    results[0].append(pred)
    results[1].append(label)
def answer(v) :
    v=np.array(v,dtype=int)
    return v
y_true,y_pred=answer(results[0]),answer(results[1])


for metric in metrics.keys() :
    print(metric+" : ",metrics[metric](y_true,y_pred))

a=np.where(y_true==15,0,1)
b=np.where(y_pred==15,0,1)
print("identification results :",np.mean(np.where(a==b,1,0)))
m2=confusion_matrix(y_true, y_pred,normalize="pred").round(2)

print("avg class : ",np.mean(np.diag(m2)))
x = ['bobcat', 'opossum', 'car', 'coyote', 'raccoon', 'bird', 'dog', 'cat', 'squirrel', 'rabbit', 'skunk', 'fox', 'rodent', 'deer',"empty"]
z_text = [[str(y) for y in x] for x in m2]


import plotly.figure_factory as ff
fig = ff.create_annotated_heatmap(m2, x=x, y=x, annotation_text=z_text, colorscale='Viridis')
fig.update_layout(
    margin=dict(t=50, l=200),
    #title="Yolo 3.0",
    xaxis_title="True labels",
    yaxis_title="Predictions",

)

fig['data'][0]['showscale'] = True
fig.show()