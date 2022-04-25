import os
import cv2 as cv
import shutil
import json


class AnimalsClassScraper:
    """
    This class is used to collect all the images which contain
    specific classes (classes that are not as well classified or detected as others,etc)
    """

    def __init__(self,dataset_path=r"data/data_split2/train/" ):
        self.dataset_path=dataset_path

        #self.to_collect_label_name = ["bird", "squirrel", "rodent"]


        self.log_dir="logs"
        if not os.path.isdir(self.log_dir) :
            os.mkdir(self.log_dir)
    def augment(self,classes= [5, 8, 12]):
        """
        Proceed to replace the image only with the ROI, now upsized to the full image size
        :param classes: The id of the classes we want to augment as integers
        :return: Nothing.
        """

        self.to_collect_label_id = classes
        count = 0
        count_weird = 0

        deleted_images_path = []
        deleted_images = []
        no_image = []

    

        for root, dirs, files in os.walk(self.dataset_path+"labels/", topdown=False):
            for file in files:
                location_file = os.path.join(root, file)
                with open(location_file) as f:
                    lines =f.readlines()
                    f.close()

                for line in lines :
                    label_id = line.split()[0]
                    bbox = line.split()[1::]  # !!!
                    if int(label_id) in self.to_collect_label_id:

                        image_name = file[:-3] + "jpg"
                        image_path = os.path.join(self.dataset_path, "images/"+image_name)
                        if os.path.exists(image_path):

                            image_data = cv.imread(image_path)
                            n=image_data.shape[0]
                            x,y,w,h=bbox
                            x,y,w,h=float(x),float(y),float(w),float(h)
                            x1=int(n*(x-w/2))
                            y1=int(n*(y-h/2))
                            x2=int(n*(x+w/2))
                            y2=int(n*(y+h/2))
                            ROI = image_data[y1:y2,x1:x2]
                            new_image_data=cv.resize(ROI,(n,n))
                            cv.imwrite(image_path,new_image_data)
                            deleted_images.append(image_name)
                            deleted_images_path.append(image_path)
                            count+= 1

                        else:
                            print(location_file, image_name)
                            no_image.append(image_name)
                            count_weird += 1


        print(count, "files haved been augmented. \n",
              count_weird, " labels with no images")

        with open(self.log_dir+"/deleted_images.txt", "w") as f:
            f.write(json.dumps(deleted_images))
        with open(self.log_dir+"/deleted_images_path.txt", "w") as f:
            f.write(json.dumps(deleted_images_path))
        with open(self.log_dir+"/no_images_path.txt", "w") as f:
            f.write(json.dumps(no_image))

    def remove(self,classes):
        """
        This function removes specific classes from the current dataset
        :param classes: list of the classes id we want to see upsampled as integers
        :return: Nothing
        """

        self.to_collect_label_id = classes
        count = 0
        count_weird = 0

        deleted_images_path = []
        deleted_images = []
        no_image = []

        for root, dirs, files in os.walk(self.dataset_path+"labels/", topdown=False):
            for file in files:
                location_file = os.path.join(root, file)
                with open(location_file) as f:
                    label_id = f.readlines()[0].split()[0]

                if int(label_id) in self.to_collect_label_id:

                    image_name = file[:-3] + "jpg"
                    image_path = os.path.join(self.dataset_path, "images/"+image_name)
                    if os.path.exists(image_path):
                        deleted_images.append(image_name)
                        deleted_images_path.append(image_path)
                        os.remove(image_path)
                        os.remove(location_file)

                        count += 1

                    else:
                        print(location_file, image_name)
                        no_image.append(image_name)
                        count_weird += 1

        print(count, "files haved been deleted. \n",
              count_weird, " labels with no images")

        with open(self.log_dir + "/deleted_images.txt", "w") as f:
            f.write(json.dumps(deleted_images))
        with open(self.log_dir + "/deleted_images_path.txt", "w") as f:
            f.write(json.dumps(deleted_images_path))
        with open(self.log_dir + "/no_images_path.txt", "w") as f:
            f.write(json.dumps(no_image))


    def upsample(self, classes,n_times):
        """

        :param self:
        :param classes: list of classes we want to see upsampled
        :param n_times:  Numbers of copy to make
        :return:  Nothing
        """

        self.to_collect_label_id = classes
        count = 0
        count_weird = 0

        deleted_images_path = []
        deleted_images = []
        no_image = []

        for root, dirs, files in os.walk(self.dataset_path+"labels/", topdown=False):
            for file in files:
                location_file = os.path.join(root, file)
                with open(location_file) as f:
                    lines = f.readlines()
                    f.close()

                for line in lines:
                    label_id = line.split()[0]

                    if int(label_id) in self.to_collect_label_id:

                        image_name = file[:-3] + "jpg"
                        image_path = os.path.join(self.dataset_path,"images/"+ image_name)
                        if os.path.exists(image_path):



                            deleted_images.append(image_name)
                            deleted_images_path.append(image_path)
                            for i in range(n_times) :
                                shutil.copy(image_path,os.path.join(self.dataset_path,"images/"+ image_name[:-4]+f"({i}).jpg"))
                            count += 1

                        else:
                            print(location_file, image_name)
                            no_image.append(image_name)
                            count_weird += 1

        print(count, "files haved been upsampled. \n",
              count_weird, " labels with no images")

        with open(self.log_dir + "/deleted_images.txt", "w") as f:
            f.write(json.dumps(deleted_images))
        with open(self.log_dir + "/deleted_images_path.txt", "w") as f:
            f.write(json.dumps(deleted_images_path))
        with open(self.log_dir + "/no_images_path.txt", "w") as f:
            f.write(json.dumps(no_image))


def main():
    scraper = AnimalsClassScraper()
    scraper.upsample([12],2)
    #scraper.remove([3])
    #scraper.augment([4])

if __name__ == "__main__":
    main()
