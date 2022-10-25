import os
import time
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.sczhedules import PiecewiseConstantDecay
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy, MeanAbsoluteError
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Mean
from PIL import Image

from datasets.loader import *
from datasets.parameters import DatasetParameters
from utils.dataset_mappings import random_crop, random_flip, random_rotate, random_lr_jpeg_noise
from utils.metrics import psnr_metric
from utils.callbacks import SaveCustomCheckpoint
from models.srresnet import build_srresnet
from models.srgan import build_discriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# HYPER PARAMETER
CROP_SIZE = 112
SCALE = 4

dataset_key = "bicubic_x4"
base_path = "/data/teamelysium/project/SRGAN/Image"
dataset_parameters = DatasetParameters(dataset_key, base_path)

train_mappings = [
    lambda lr, hr: random_crop(lr, hr, hr_crop_size=CROP_SIZE, scale=dataset_parameters.scale),
                   random_flip,
                   random_rotate,
                   random_lr_jpeg_noise
]

train_dataset, valid_dataset = create_train_valid_datasets(dataset_parameters, train_mappings)
valid_dataset_subset = valid_dataset.take(10)

generator = build_srresnet(scale=dataset_parameters.scale)
print(generator.summary())

ckpt_dir = f"./ckpt/sr_resnet_{dataset_key}"
learning_rate = 1e-4
ckpt = tf.train.Checkpoint(step=tf.Variable(0),
                           epoch=tf.Variable(0),
                           psnr=tf.Variable(0.0),
                           optimizer=Adam(learning_rate),
                           model=generator)

ckpt_manager = tf.train.CheckpointManager(checkpoint=ckpt,
                                          directory=ckpt_dir,
                                          max_to_keep=3)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f'Model restored from checkpoint at step {ckpt.step.numpy()} with validation PSNR {ckpt.psnr.numpy()}.')

training_steps = 1_000_000

steps_per_epoch = 1000

training_epochs = training_steps / steps_per_epoch

if ckpt.epoch.numpy() < training_epochs:
    remaining_epochs = int(training_epochs - ckpt.epoch.numpy())
    print(f"Continuing Training from epoch {ckpt.epoch.numpy()}. Remaining epochs: {remaining_epochs}.")
    save_checkpoint_callback = SaveCustomCheckpoint(ckpt_manager, steps_per_epoch)
    ckpt.model.compile(optimizer=ckpt.optimizer, loss=MeanSquaredError(), metrics=[psnr_metric])
    ckpt.model.fit(train_dataset,validation_data=valid_dataset_subset, steps_per_epoch=steps_per_epoch, epochs=remaining_epochs, callbacks=[save_checkpoint_callback])
else:
    print("Training already completed. To continue training, increase the number of training steps")

weights_directory = f"weights/srresnet_{dataset_key}"
os.makedirs(weights_directory, exist_ok=True)
weights_file = f'{weights_directory}/generator.h5'
ckpt.model.save_weights(weights_file)
print(f"[{weights_file}]ckpt saved..!")

generator = build_srresnet(scale=dataset_parameters.scale)
generator.load_weights(weights_file)
discriminator = build_discriminator(hr_crop_size=CROP_SIZE)

layer_5_4 = 20
vgg = VGG19(input_shape=(None, None, 3), include_top=False)
perceptual_model = Model(vgg.input, vgg.layers[layer_5_4].output)

binary_cross_entropy = BinaryCrossentropy()
mean_squared_error = MeanSquaredError()

learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000], values=[1e-4, 1e-5])

generator_optimizer = Adam(learning_rate=learning_rate)
discriminator_optimizer = Adam(learning_rate=learning_rate)

srgan_checkpoint_dir=f'./ckpt/srgan_{dataset_key}'

srgan_checkpoint = tf.train.Checkpoint(step=tf.Variable(0),
                                       psnr=tf.Variable(0.0),
                                       generator_optimizer=Adam(learning_rate),
                                       discriminator_optimizer=Adam(learning_rate),
                                       generator=generator,
                                       discriminator=discriminator)

srgan_checkpoint_manager = tf.train.CheckpointManager(checkpoint=srgan_checkpoint,
                                                directory=srgan_checkpoint_dir,
                                                max_to_keep=3)

if srgan_checkpoint_manager.latest_checkpoint:
    srgan_checkpoint.restore(srgan_checkpoint_manager.latest_checkpoint)
    print(f'Model restored from checkpoint at step {srgan_checkpoint.step.numpy()} with validation PSNR {srgan_checkpoint.psnr.numpy()}.')

@tf.function
def train_step(lr, hr):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        lr = tf.cast(lr, tf.float32)
        hr = tf.cast(hr, tf.float32)

        sr = srgan_checkpoint.generator(lr, training=True)

        hr_output = srgan_checkpoint.discriminator(hr, training=True)
        sr_output = srgan_checkpoint.discriminator(sr, training=True)

        con_loss = calculate_content_loss(hr, sr)
        gen_loss = calculate_generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = calculate_discriminator_loss(hr_output, sr_output)

    gradients_of_generator = gen_tape.gradient(perc_loss, srgan_checkpoint.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, srgan_checkpoint.discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, srgan_checkpoint.generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, srgan_checkpoint.discriminator.trainable_variables))

    return perc_loss, disc_loss

@tf.function
def calculate_content_loss(hr, sr):
    sr = preprocess_input(sr)
    hr = preprocess_input(hr)
    sr_features = perceptual_model(sr) / 12.75
    hr_features = perceptual_model(hr) / 12.75
    return mean_squared_error(hr_features, sr_features)

def calculate_generator_loss(sr_out):
    return binary_cross_entropy(tf.ones_like(sr_out), sr_out)

def calculate_discriminator_loss(hr_out, sr_out):
    hr_loss = binary_cross_entropy(tf.ones_like(hr_out), hr_out)
    sr_loss = binary_cross_entropy(tf.zeros_like(sr_out), sr_out)
    return hr_loss + sr_loss

perceptual_loss_metric = Mean()
discriminator_loss_metric = Mean()

step = srgan_checkpoint.step.numpy()
steps = 200000

monitor_folder = f"monitor_training/srgan_{dataset_key}"
os.makedirs(monitor_folder, exist_ok=True)

now = time.perf_counter()

for lr, hr in train_dataset.take(steps - step):
    srgan_checkpoint.step.assign_add(1)
    step = srgan_checkpoint.step.numpy()

    perceptual_loss, discriminator_loss = train_step(lr, hr)
    perceptual_loss_metric(perceptual_loss)
    discriminator_loss_metric(discriminator_loss)

    if step % 1000 == 0:
        psnr_values = []
        
        for lr, hr in valid_dataset_subset:
            sr = srgan_checkpoint.generator.predict(lr)[0]
            sr = tf.clip_by_value(sr, 0, 255)
            sr = tf.round(sr)
            sr = tf.cast(sr, tf.uint8)
            
            psnr_value = psnr_metric(hr, sr)[0]
            psnr_values.append(psnr_value)
            psnr = tf.reduce_mean(psnr_values)
            
        image = Image.fromarray(sr.numpy())
        image.save(f"{monitor_folder}/{step}.png" )
        
        duration = time.perf_counter() - now
        
        now = time.perf_counter()
        
        print(f'{step}/{steps}, psnr = {psnr}, perceptual loss = {perceptual_loss_metric.result():.4f}, discriminator loss = {discriminator_loss_metric.result():.4f} ({duration:.2f}s)')
        
        perceptual_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()
        
        srgan_checkpoint.psnr.assign(psnr)
        srgan_checkpoint_manager.save()

weights_directory = f"weights/srgan_{dataset_key}"
os.makedirs(weights_directory, exist_ok=True)
weights_file = f'{weights_directory}/generator.h5'
srgan_checkpoint.generator.save_weights(weights_file)
