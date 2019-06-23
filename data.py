
import os
import numpy as np
from sklearn import model_selection
from matplotlib import pyplot as plt

def load_from_path(path):
    imgs = os.listdir(path)
    images = []
    for img in imgs:
        image = plt.imread(os.path.join(path, img))[:,:,:3]
        images.append(image)
    return np.array(images)

def split_into_train_test(images):
    train, test = model_selection.train_test_split(images, train_size=0.8)
    return train, test



