import os
import numpy as np
from sklearn.metrics import confusion_matrix
num_classes=15
# ------------defining metrics--------------------------------------------

#------------defining metrics--------------------------------------------
import sklearn





def f1(true,pred) :

    return sklearn.metrics.f1_score(true,pred,average='weighted') #weighted??

def precision(true,pred):

    return sklearn.metrics.precision_score(true,pred,average='macro')

def recall(true,pred):

    return sklearn.metrics.recall_score(true,pred,average='macro')



metrics={
    "f1"    :    f1,
    "precision": precision,
    "recall":    recall,

}





def yolo_testing(exp_folder,label_folder) :
    """

    :param exp_folder: the folder of the ouptut labels of yolo's detect.py code
    :param label_folder: The folder of the true labels
    :return: A concatenated tensor of the predicted classes and true classes
    """
    mapping = {}
    for file in os.listdir(label_folder):
        file_id = file.split("_")[0]
        mapping[file_id] = file

    results = [[], []]

    multicount = False




    for file in os.listdir(label_folder) :



        if os.path.getsize(label_folder+"/"+file)>1 :
            true_label=np.loadtxt(label_folder+"/"+file,unpack=True)
            true_label = true_label.flatten()[0]
        else :
            true_label=14

        if os.path.exists(exp_folder + "/" + file) :
            label = np.loadtxt(exp_folder + "/" + file, unpack=True)
            label=label.flatten()[0]
        else :
            label=14

        results[0].append(int(true_label))
        results[1].append(int(label))

    return results


def plot(results) :
    def answer(v):
        v = np.array(v, dtype=int)
        return v

    y_true, y_pred = answer(results[0]), answer(results[1])
    print(len(results),y_true.shape,y_pred.shape )
    for metric in metrics.keys():
        print(metric + " : ", metrics[metric](y_true, y_pred))

    a = np.where(y_true == 14, 0, 1)
    b = np.where(y_pred == 14, 0, 1)
    print("identification results :", np.mean(np.where(a == b, 1, 0)))
    m2 = confusion_matrix(y_true, y_pred, normalize="pred").round(2)
    #np.savetxt("yolo_unseen_confusion.txt", m2)
    print("avg class : ", np.mean(np.diag(m2)))
    print("top-1 :",np.mean(np.where(y_true==y_pred,1,0)))
    x = ['bobcat', 'opossum', 'car', 'coyote', 'raccoon', 'bird', 'dog', 'cat', 'squirrel', 'rabbit', 'skunk', 'fox',
         'rodent', 'deer', "empty"]
    print(m2.shape)
    z_text = [[str(y) for y in x] for x in m2]

    import plotly.figure_factory as ff
    fig = ff.create_annotated_heatmap(m2, x=x, y=x, annotation_text=z_text, colorscale="Blues")
    fig.update_layout(
        margin=dict(t=50, l=200),
        # title="Yolo 3.0",
        xaxis_title="True labels",
        yaxis_title="Predictions",

    )

    fig['data'][0]['showscale'] = True
    import plotly.io as pio
    pio.write_image(fig, 'yolo_confused.png', width=1920, height=1080)
    fig.show()


if __name__=="__main__" :

    results=yolo_testing("models/yolov5/runs/detect/exp/labels","data/test_set3/test/labels")
    plot(results)