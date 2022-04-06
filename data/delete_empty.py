import os


class DeleteEmpty:
    """
    This class is used to delete all the .txt file which contain the label "empty"
    (id=30 or 14 depending on the dataset version).
    Our basic dataset contains these by default but we need to delete
    these .txt files in order to feed yolo wiht the dataset.
    """

    def __init__(self, ):
        self.train_path = f"./data/data_split2/train/labels/"
        self.valid_path = f"./data/data_split2/valid/labels/"
        self.test_path = f"./data/data_split2/test/labels/"
        self.to_delete_label = "empty"
        self.to_delete_id = 30

    def delete(self, ):
        datasets = {'train': self.train_path, 'valid': self.valid_path, 'test': self.test_path}
        count = 0
        for dataset_name, dataset_path in datasets.items():
            for root, dirs, files in os.walk(dataset_path, topdown=False):
                for file in files:
                    location_file = os.path.join(root, file)
                    with open(location_file) as f:
                        label_id = f.readlines()[0].split()[0]
                        f.close()

                    if int(label_id) == self.to_delete_id:
                        if os.path.exists(location_file):
                            os.remove(location_file)
                            count += 1
                        else:
                            print("path doesn't exist")
        print(count, "txt file have been deleted.")


def main():
    delete_empty = DeleteEmpty()
    delete_empty.delete()


if __name__ == "__main__":
    main()
