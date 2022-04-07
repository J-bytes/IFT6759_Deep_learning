# deleted_images_path = []
    # deleted_images = []
# Find .txt files with specific labels
# copy paste the corresponding images into a new external folder (containing 3 subfolders, train-valid-test)
# store the name of these images
    # ADD CORRESPONDING IMAGE TO SPECIFIC FOLDER
    # deleted_images_path.append(location_file)
    # deleted_images.append(file)
# store the count of selected images per class
# delete these images and annotations from the dataset
    # if os.path.exists("demofile.txt"):
    #os.remove("demofile.txt")  # one file at a time
import os
import cv2 as cv
import shutil
import json
class AnimalsClassScraper:
    """
    This class is used to collect all the images which contain
    specific classes (classes that are not as well classified or detected as others),
    and delete them (and their corresponding annotation) from the dataset.
    Once the images are collected and placed into a new folder named "to_augment",
    we augment the images using an external tool.
    We then merge the original dataset with the new augmented images and run the next experiment.
    """
    def __init__(self, ):
        self.train_label_path = r"C:\Users\cdetr\Documents\GitHub\UdeM\selim_project\IFT6759_Deep_learning\data\data_augmented\train\labels"
        self.valid_label_path = r"C:\Users\cdetr\Documents\GitHub\UdeM\selim_project\IFT6759_Deep_learning\data\data_augmented\valid\labels"
        self.test_label_path = r"C:\Users\cdetr\Documents\GitHub\UdeM\selim_project\IFT6759_Deep_learning\data\data_augmented\test\labels"
        self.train_image_path = r"C:\Users\cdetr\Documents\GitHub\UdeM\selim_project\IFT6759_Deep_learning\data\data_augmented\train\images"
        self.valid_image_path = r"C:\Users\cdetr\Documents\GitHub\UdeM\selim_project\IFT6759_Deep_learning\data\data_augmented\valid\images"
        self.test_image_path = r"C:\Users\cdetr\Documents\GitHub\UdeM\selim_project\IFT6759_Deep_learning\data\data_augmented\test\images"

        self.to_collect_label_name = ["bird", "squirrel", "rodent"]
        self.to_collect_label_id = [5, 8, 12]


    def collect(self, ):
        datasets_labels = {'train': self.train_label_path, 'valid': self.valid_label_path, 'test': self.test_label_path}
        datasets_images = {'train': self.train_image_path, 'valid': self.valid_image_path, 'test': self.test_image_path}

        count_train = 0
        count_valid = 0
        count_test = 0
        count_weird = 0

        deleted_images_path = []
        deleted_images = []
        no_image = []

        for dataset_name, dataset_path in datasets_labels.items():

            for root, dirs, files in os.walk(dataset_path, topdown=False):
                for file in files:
                    location_file = os.path.join(root, file)
                    with open(location_file) as f:
                        label_id = f.readlines()[0].split()[0]
                        f.close()
                    if int(label_id) in self.to_collect_label_id:
                        filename = location_file[-40:-3]
                        image_name = location_file[-40:-3] +'jpg'
                        image_path = os.path.join(datasets_images[dataset_name],image_name)

                        #print(image_path)
                        if not os.path.isdir(f"data/data_split2/" + dataset_name +"_to_augment/" ):
                           os.mkdir(f"data/data_split2/" + dataset_name +"_to_augment/")
                           os.mkdir(f"data/data_split2/" + dataset_name + "_to_augment/images")
                           os.mkdir(f"data/data_split2/" + dataset_name + "_to_augment/labels/")
                        if os.path.exists(image_path):

                            image_data = cv.imread(image_path)
                            #print(image_path)
                            cv.imwrite(f"data/data_split2/" + dataset_name +"_to_augment/images/" + image_name, image_data)
                            shutil.copy(location_file, f"data/data_split2/" + dataset_name +"_to_augment/labels/" + filename +"txt")
                            deleted_images.append(image_name)
                            deleted_images_path.append(image_path)
                            os.remove(location_file)
                            os.remove(image_path)
                            if dataset_name == 'train':
                                count_train += 1
                            if dataset_name == 'valid':
                                count_valid += 1
                            if dataset_name == 'test':
                                count_test += 1
                        else :
                            print(location_file,image_name)
                            no_image.append(image_name)
                            count_weird += 1
        with open('deleted_images.txt', 'w') as f:
            f.write(json.dumps(deleted_images))
        with open('deleted_images_path.txt', 'w') as f:
            f.write(json.dumps(deleted_images_path))
        with open('no_images_path.txt', 'w') as f:
            f.write(json.dumps(no_image))

        print(count_train, "files haved been moved from train.","\n",
              count_valid, "files haved been moved from valid.","\n",
              count_test, "files haved been moved from test.","\n",
              count_weird, " labels with no images")
def main():
    scraper = AnimalsClassScraper()
    scraper.collect()

if __name__ == "__main__":
    main()