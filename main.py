import os
import argparse

import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import datasets, optimizers
import numpy as np

from network import VGG16_A, VGG16_B, VGG16_C, VGG16_D, VGG16_E

tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def normalize(X_train, X_test):
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def prepare_cifar(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', type=str, required=True)
    p.add_argument('--dropout_ratio', type=float, default=.0) 
    config = p.parse_args()
    return config

def get_model(config):
    if config.model_fn == 'VGG16_A':
      model = VGG16_A(dropout_ratio = config.dropout_ratio)
    elif config.model_fn == 'VGG16_B': 
      model = VGG16_B(dropout_ratio = config.dropout_ratio)
    elif config.model_fn == 'VGG16_C':
      model = VGG16_C(dropout_ratio = config.dropout_ratio)
    elif config.model_fn == 'VGG16_D':
      model = VGG16_D(dropout_ratio = config.dropout_ratio)
    elif config.model_fn == 'VGG16_E':
      model = VGG16_E(dropout_ratio = config.dropout_ratio)
    else:
        raise NotImplementedError('You need to specify model name.')

    return model

def main(config):

    print('loading data...')
    (x,y), (x_test, y_test) = datasets.cifar10.load_data()
    x, x_test = normalize(x, x_test)
    print(x.shape, y.shape, x_test.shape, y_test.shape)

    train_loader = tf.data.Dataset.from_tensor_slices((x,y))
    train_loader = train_loader.map(prepare_cifar).shuffle(50000).batch(256)

    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_loader = test_loader.map(prepare_cifar).shuffle(10000).batch(256)
    print('done.')

    model = get_model(config)
    model.build(input_shape=(None, 32, 32, 3))

    model.summary()

    loss_fn = keras.losses.CategoricalCrossentropy()
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)


    for epoch in range(100):

        for step, (x, y) in enumerate(train_loader):

            y = tf.one_hot(tf.squeeze(y, axis=1), depth=10)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = loss_fn(y, logits)    
                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 40 == 0:
                print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
                metric.reset_states()


        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in test_loader:
                y = tf.one_hot(tf.squeeze(y, axis=1), depth=10)

                logits = model.predict(x)

                metric.update_state(y, logits)
            print('test acc:', metric.result().numpy())
            metric.reset_states()


if __name__ == '__main__':
    config = define_argparser()
    main(config)