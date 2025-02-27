import os
import numpy as np


class MNISTDataParser:
    def __init__(self, dataset_dir='data/MNIST', normalize='none'):
        self.dataset_dir = dataset_dir
        self.normalize = normalize
        self.images = None
        self.labels = None

    def parse(self):
        train_image_path = os.path.join(self.dataset_dir, 'train-images.idx3-ubyte')
        train_labels_path = os.path.join(self.dataset_dir, 'train-labels.idx1-ubyte')
        test_image_path = os.path.join(self.dataset_dir, 't10k-images.idx3-ubyte')
        test_labels_path = os.path.join(self.dataset_dir, 't10k-labels.idx1-ubyte')

        with open(train_image_path, 'rb') as file:
            image_train = np.frombuffer(file.read(), dtype=np.uint8)
            image_train = image_train[16:].reshape(-1, 28, 28)

        with open(train_labels_path, 'rb') as file:
            labels_train = np.frombuffer(file.read(), dtype=np.uint8)
            labels_train = labels_train[8:]

        with open(test_image_path, 'rb') as file:
            image_test = np.frombuffer(file.read(), dtype=np.uint8)
            image_test = image_test[16:].reshape(-1, 28, 28)

        with open(test_labels_path, 'rb') as file:
            labels_test = np.frombuffer(file.read(), dtype=np.uint8)
            labels_test = labels_test[8:]

        self.images = np.vstack((image_train, image_test))
        self.labels = np.hstack((labels_train, labels_test))

        if self.normalize == 'minmax':
            self.images = self.minmax_normalize(self.images)
        elif self.normalize == 'standard':
            self.images = self.standardize(self.images)

        print(f'Labels shape: {self.labels.shape}')
        print(f'Images shape: {self.images.shape}')

    @staticmethod
    def minmax_normalize(self, images):
        images_flat = images.reshape(images.shape[0], -1)
        images_normalized = images_flat / 255.0
        return images_normalized.reshape(images.shape)

    @staticmethod
    def standardize(self, images):
        images_flat = images.reshape(images.shape[0], -1)
        images_normalized = images_flat / 255.0
        mean = np.mean(images_normalized, axis=0)
        std = np.std(images_normalized, axis=0)
        std = np.where(std == 0, 1e-8, std)
        images_standardized = (images_normalized - mean) / std
        return images_standardized.reshape(images.shape)

# Example usage:
parser = MNISTDataParser(normalize='standard')
parser.parse()
