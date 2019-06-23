
import tensorflow as tf
import numpy as np

class PreProcessing():
    def __init__(self,
        downscale_factor=2):
        self.downscale_factor = downscale_factor

    def get_spatial_info(self, images):
        if len(images.shape) <= 3:
            return images.shape[0], images.shape[1]
        else:
            return images.shape[1], images.shape[2]

    def convert_high_resolution_to_low_resolution(self, images):
        x_axis, y_axis = self.get_spatial_info(images)
        x_axis = x_axis // self.downscale_factor
        y_axis = y_axis // self.downscale_factor
        return tf.image.resize(images, (x_axis, y_axis), 
            method=tf.image.ResizeMethod.BICUBIC)

    def normalize_images(self, images):
        return (images.astype(np.float32) - 127.5) / 127.5

    def unnormalize_images(self, images):
        return ((images + 1) * 127.5).astype(np.uint8)



