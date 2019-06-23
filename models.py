
import tensorflow as tf


def get_vgg():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    return tf.keras.models.Model(vgg.input, vgg.get_layer('block5_conv4').output)

class ConvBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides):
        super(ConvBlock, self).__init__()
        self.Conv2D = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")
        self.BN = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    def call(self, x):
        x = self.Conv2D(x)
        x = self.BN(x)
        x = self.activation(x)
        return x

class ResidualBlock(tf.keras.Model):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.BN1 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.activation = tf.keras.layers.PReLU()
        self.Conv2D_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding = "same")
        self.BN2 = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.add = tf.keras.layers.Add()
    
    def call(self, x):
        x1 = self.Conv2D_1(x)
        x1 = self.BN1(x1)
        x1 = self.activation(x1)
        x1 = self.Conv2D_2(x1)
        x1 = self.BN2(x1)
        return self.add([x, x1])

class UpSamplingBlock(tf.keras.Model):
    def __init__(self):
        super(UpSamplingBlock, self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same")
        self.Upsample = tf.keras.layers.UpSampling2D(size=2)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)
    
    def call(self, x):
        x = self.Conv2D_1(x)
        x = self.Upsample(x)
        x = self.activation(x)
        return x

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.Conv2D_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=9, strides=1, padding="same")
        self.activation_1 = tf.keras.layers.PReLU()
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.res_block_4 = ResidualBlock()
        self.res_block_5 = ResidualBlock()
        self.res_block_6 = ResidualBlock()
        self.Conv2D_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.BN = tf.keras.layers.BatchNormalization(momentum=0.5)
        self.add = tf.keras.layers.Add()
        self.upsample_block_1 = UpSamplingBlock()
        self.upsample_block_2 = UpSamplingBlock()
        self.Conv2D_3 = tf.keras.layers.Conv2D(filters=3, kernel_size=9, strides=1, padding="same")
    
    def call(self, x): 
        x1 = self.Conv2D_1(x)
        x1 = self.activation_1(x1)
        x2 = self.res_block_1(x1)
        x2 = self.res_block_2(x2)
        x2 = self.res_block_3(x2)
        x2 = self.res_block_4(x2)
        x2 = self.res_block_5(x2)
        x2 = self.res_block_6(x2)
        x3 = self.Conv2D_2(x2)
        x3 = self.BN(x3)
        x4 = self.add([x1, x3])
        x4 = self.upsample_block_1(x4)
        x4 = self.upsample_block_2(x4)
        x4 = self.Conv2D_3(x4)
        return tf.keras.activations.tanh(x4)

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.Conv2D = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.activation_1 = tf.keras.layers.LeakyReLU(alpha=0.2)       
        self.ConvBlock_1 = ConvBlock(64, 3, 2)
        self.ConvBlock_2 = ConvBlock(128, 3, 1)
        self.ConvBlock_3 = ConvBlock(128, 3, 2)
        self.ConvBlock_4 = ConvBlock(256, 3, 1)
        self.ConvBlock_5 = ConvBlock(256, 3, 2)
        self.ConvBlock_6 = ConvBlock(512, 3, 1)
        self.ConvBlock_7 = ConvBlock(512, 3, 2)        
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.activation_2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.dense_2 = tf.keras.layers.Dense(1)

    def call(self, x): 
        x = self.Conv2D(x)
        x = self.activation_1(x)
        x = self.ConvBlock_1(x)
        x = self.ConvBlock_2(x)
        x = self.ConvBlock_3(x)
        x = self.ConvBlock_4(x)
        x = self.ConvBlock_5(x)
        x = self.ConvBlock_6(x)
        x = self.ConvBlock_7(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.activation_2(x)
        x = self.dense_2(x)
        x = tf.keras.activations.sigmoid(x)
        return x