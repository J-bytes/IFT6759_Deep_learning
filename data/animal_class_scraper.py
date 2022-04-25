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

    def __init__(
        self,
    ):
        self.train_label_path = r"data/data_split2/train/labels"
        self.valid_label_path =  r"data/data_split2/valid/labels"
        self.test_label_path =  r"data/data_split2/test/labels"
        self.train_image_path = r"data/data_split2/train/images"
        self.valid_image_path =  r"data/data_split2/valid/images"
        self.test_image_path =  r"data/data_split2/test/images"

        self.to_collect_label_name = ["bird", "squirrel", "rodent"]
        self.to_collect_label_id = [5, 8, 12]


    def collect(self, ):
        datasets_labels = {'train': self.train_label_path}#, 'valid': self.valid_label_path, 'test': self.test_label_path}
        datasets_images = {'train': self.train_image_path}#, 'valid': self.valid_image_path, 'test': self.test_image_path}

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
                        lines =f.readlines()
                        f.close()

                    for line in lines :
                        label_id = line.split()[0]
                        bbox = line.split()[1::]  # !!!
                        if int(label_id) in self.to_collect_label_id:

                            image_name = location_file[-40:-3] + "jpg"
                            image_path = os.path.join(
                                datasets_images[dataset_name], image_name
                            )

                            # print(image_path)
                            if not os.path.isdir(
                                f"data/data_split2/" + dataset_name + "_to_augment/"
                            ):
                                os.mkdir(
                                    f"data/data_split2/" + dataset_name + "_to_augment/"
                                )
                                os.mkdir(
                                    f"data/data_split2/"
                                    + dataset_name
                                    + "_to_augment/images"
                                )
                                os.mkdir(
                                    f"data/data_split2/"
                                    + dataset_name
                                    + "_to_augment/labels/"
                                )
                            if os.path.exists(image_path):

                                image_data = cv.imread(image_path)
                                # print(image_path)
                                # cv.imwrite(
                                #     f"data/data_split2/"
                                #     + dataset_name
                                #     + "_to_augment/images/"
                                #     + image_name,
                                #     image_data,
                                # )
                                # shutil.copy(
                                #     location_file,
                                #     f"data/data_split2/"
                                #     + dataset_name
                                #     + "_to_augment/labels/"
                                #     + filename
                                #     + "txt",
                                # )
                                n=image_data.shape[0]
                                x,y,w,h=bbox
                                x,y,w,h=float(x),float(y),float(w),float(h)
                                x1=int(n*(x-w/2))
                                y1=int(n*(y-w/2))
                                x2=int(n*(x+w/2))
                                y2=int(n*(y+w/2))
                                ROI = image_data[y1:y2,x1:x2]
                                new_image_data=cv.resize(ROI,(320,320))
                                cv.imwrite(
                                    f"data/data_split2/"
                                    + dataset_name
                                    + "images/"
                                    + image_name,
                                    new_image_data,
                                )

                                deleted_images.append(image_name)
                                deleted_images_path.append(image_path)
                                #os.remove(location_file)
                                os.remove(image_path)
                                if dataset_name == "train":
                                    count_train += 1
                                if dataset_name == "valid":
                                    count_valid += 1
                                if dataset_name == "test":
                                    count_test += 1
                            else:
                                print(location_file, image_name)
                                no_image.append(image_name)
                                count_weird += 1
        with open("deleted_images.txt", "w") as f:
            f.write(json.dumps(deleted_images))
        with open("deleted_images_path.txt", "w") as f:
            f.write(json.dumps(deleted_images_path))
        with open("no_images_path.txt", "w") as f:
            f.write(json.dumps(no_image))

        print(
            count_train,
            "files haved been moved from train.",
            "\n",
            count_valid,
            "files haved been moved from valid.",
            "\n",
            count_test,
            "files haved been moved from test.",
            "\n",
            count_weird,
            " labels with no images",
        )


        print(count_train, "files haved been moved from train.","\n",
              #count_valid, "files haved been moved from valid.","\n",
              #count_test, "files haved been moved from test.","\n",
              count_weird, " labels with no images")
def main():
    scraper = AnimalsClassScraper()
    scraper.collect()


if __name__ == "__main__":
    main()
