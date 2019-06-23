

from models import get_vgg, Generator, Discriminator
from utils import PreProcessing
from data import load_from_path, split_into_train_test
import tensorflow as tf
import os

generator = Generator()
discriminator = Discriminator()
vgg = get_vgg()

generator_optimizer = tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
discriminator_optimizer = tf.keras.optimizers.Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)



checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def discriminator_loss(hr, sr):
    real_loss = -tf.reduce_mean(tf.multiply(tf.ones_like(hr), tf.math.log(hr)) + tf.multiply((1-tf.ones_like(hr)), tf.math.log(1-hr)))
    generated_loss = -tf.reduce_mean(tf.multiply(tf.ones_like(sr), tf.math.log(sr)) + tf.multiply((1-tf.ones_like(sr)), tf.math.log(1-sr)))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

def vgg_loss(pred, real):
    pred_value = vgg(pred)
    real_value = vgg(real)
    return tf.losses.mean_squared_error(real_value, pred_value)

@tf.function
def train_step(hr, lr):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr = generator(lr)
        disc_real = discriminator(hr)
        disc_fake = discriminator(sr)
        gen_loss = vgg_loss(sr, hr)
        disc_loss = discriminator_loss(disc_real, disc_fake)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))


def train(train_hr, train_lr, 
    test_hr, test_lr, epochs=100):
    for epoch in range(epochs):
        for tr_hr, tr_lr in zip(train_hr, train_lr):
            tr_hr = tf.expand_dims(tr_hr, axis=0)
            tr_lr = tf.expand_dims(tr_lr, axis=0)
            train_step(tr_hr, tr_lr)
            print("Train Step Completed")

        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


print("Loading Images")
images = load_from_path('./dataset/')
print("Images Loaded from Path")

pre_processing = PreProcessing(downscale_factor=4)

images = pre_processing.normalize_images(images)
print("Images Normalized")

train_hr, test_hr = split_into_train_test(images)
print("Divided into Train Test")

train_lr = pre_processing.convert_high_resolution_to_low_resolution(train_hr)
test_lr = pre_processing.convert_high_resolution_to_low_resolution(test_hr)
print("Low Res images Created")

print("Started Training")
train(train_hr, train_lr, test_hr, test_lr)
