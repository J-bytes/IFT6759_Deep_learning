import os
import numpy as np
import pandas as pd
import json
import shutil
import matplotlib.pyplot as plt

used_locations=os.listdir("data/images")

mapping={
    "bobcat" :0,
    "opossum":1,
     "car": 2,
    "coyote" :3,
    "raccoon" :4,
    "bird" : 5,
    "dog" : 6,
    "cat" : 7,
    "squirrel" :8,
    "rabbit" :9,
    "skunk":10,
    "fox" :11,
    "rodent":12,
    "deer":13,
    "empty" :14,
}

used_category_names=["opossum"]#mapping.keys()
image_count={}
for i in used_category_names :
    image_count[i]=0


data=json.load(open("caltech_images_20210113.json"))
data_dir="data"
new_data_dir="data/test_set2/test"
# format data better cuz they did a shit job
annotations = {}
for file in data["annotations"]:
    annotations[file["image_id"]] = file

categories = {}
for item in data["categories"]:
    categories[item["id"]] = item["name"]

def part1() :
    if not os.path.isdir(new_data_dir) :
        os.mkdir(new_data_dir)
        os.mkdir(new_data_dir+"/images")
        os.mkdir(new_data_dir+"/labels")

    for file in data["images"]:

        if file["location"] not in used_locations : # new location

            category_id=annotations[file["id"]]["category_id"]

            if categories[category_id] in used_category_names :
                #add this image to the test set
                if image_count[categories[category_id]]<100 :
                    shutil.copy(f"{data_dir}/images/{file['id']}.jpg",f"{new_data_dir}/images/{file['id']}.jpg")
                    image_count[categories[category_id]]+=1


mapping2={

}

def part2() :
    #assuming roboflow annotation has been done and pushed to test/labels
    count=0
    labels={}
    not_empty="not_empty"
    should_empty="should_empty"
    if not os.path.isdir(not_empty) :
        os.mkdir(not_empty)
    if not os.path.isdir(should_empty) :
        os.mkdir(should_empty)
    for file in os.listdir(new_data_dir+"/images") :




        file_id=file.split("_")[0]
        annot = np.loadtxt(f"{new_data_dir}/labels/{file[:-4]}.txt")
        if len(annot)!=0 :

            num_lines = sum(1 for line in open(f"{new_data_dir}/labels/{file[:-4]}.txt"))
            annot=annot.reshape(num_lines,5)
            os.remove(f"{new_data_dir}/labels/{file[:-4]}.txt")
            with open(f"{new_data_dir}/labels/{file[:-4]}.txt","a") as f :
                for ex,_ in enumerate(annot) :
                    category_id, new_x, new_y, new_width, new_height=annot[ex]
                    id=mapping[categories[annotations[file_id]["category_id"]]]
                    if id==14 :
                        print(f"The file is said to be empty!. \n {file}")
                        #shutil.copy(new_data_dir + "/images/" + file, should_empty + "/" + file)
                        os.remove(new_data_dir + "/images/" + file)
                        os.remove(new_data_dir + "/labels/" + file[:-4]+".txt")
                        count+=1
                    annot[ex,0]=id


                #np.savetxt(f"{new_data_dir}/labels/{file[:-4]}.txt",annot)

                    to_write = str(
                        str(int(category_id)) + ' ' + str(new_x) + ' ' + str(new_y) + ' ' + str(new_width) + ' ' + str(new_height))

                    f.write(to_write)

        else :
            if mapping[categories[annotations[file_id]["category_id"]]] !=14 :
                print(f"The file has no label and yet should not be empty. \n {file}")
                #shutil.copy(new_data_dir+"/images/"+file,not_empty+"/"+file)
                os.remove(new_data_dir+"/images/"+file)
                os.remove(new_data_dir+"/labels/"+file[:-4]+".txt")
                count+=1
                labels[file]=categories[annotations[file_id]["category_id"]]

    f= open("should_empty_labels.json","w+")
    json.dump(labels,f)
    f.close()
    print("count :",count)

#part2()


def part3() : # lets plot a histogram to confirm everything is A-ok
    data={}
    for file in os.listdir(new_data_dir + "/images"):

        file_id=file.split("_")[0]
        data[categories[annotations[file_id]["category_id"]]]=data.get(categories[annotations[file_id]["category_id"]],0)+1

    data = pd.DataFrame(data.items())
    data.plot(kind="bar")
    plt.xticks(rotation=45, fontsize=25)
    plt.xlabel("Classes")  # , fontsize = 60)
    plt.ylabel("Count")  # , fontsize = 60)
    plt.legend()  # prop={'size':45})
    # plt.title("Distribution of classes in the differents datasets", fontsize = 30)
    plt.savefig("histogram_test_set.png")

part2()
part3()