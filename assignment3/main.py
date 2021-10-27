import os
import argparse

import numpy as np 
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import datasets, optimizers
from tensorflow.python.keras.layers.recurrent import LSTM, RNN
from network import LSTM_FC, RNN_FC
from tqdm import tqdm

plt.rcParams['figure.figsize'] = (10.0, 8.0)

tf.random.set_seed(42)
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', type=str, required=True)
    p.add_argument('--num_classes', type=int, default=2) 
    p.add_argument('--max_features', type=int, default=20000) 
    p.add_argument('--max_len', type=int, default=200) 
    p.add_argument('--units', type=int, default=64) 
    config = p.parse_args()
    return config

def get_model(config):
    if config.model_fn == 'LSTM_FC':
      model = LSTM_FC(num_classes=config.num_classes, 
                      max_features=config.max_features,
                      max_len=config.max_len,
                      units=config.units)
    elif config.model_fn == 'RNN_FC': 
      model = RNN_FC(num_classes=config.num_classes, 
                     max_features=config.max_features,
                     max_len=config.max_len,
                     units=config.units)
    else:
        raise NotImplementedError('You need to specify model name.')

    return model

def main(config):

    print('loading data...')
    (x_train, y_train), (x_val, y_val) = datasets.imdb.load_data(
        num_words=config.max_features
    )

    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=config.max_len)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=config.max_len)

    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")

    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_loader = train_loader.shuffle(25000).batch(32)

    test_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_loader = test_loader.shuffle(25000).batch(32)
    print('done.')
    

    print(x_train.shape)

    model = get_model(config)
    model.build(input_shape=(None, 200))

    model.summary()

    loss_fn = keras.losses.CategoricalCrossentropy()
    metric = keras.metrics.CategoricalAccuracy()

    optimizer = optimizers.Adam(learning_rate=0.0001)

    train_acc, train_loss, test_acc = [], [], []

    for epoch in range(1):

        for step, (x, y) in enumerate(tqdm(train_loader)):
            y = tf.one_hot(y, depth=2)

            with tf.GradientTape() as tape:
                logits = model(x)
                loss = loss_fn(y, logits)    
                metric.update_state(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 == 0:
              tqdm.write(f"EPOCH: {epoch+1} STEP: {step} loss: {float(loss):.2f} acc: {metric.result().numpy():.2f}")
              metric.reset_states()
        
        train_loss.append(loss.numpy())
        train_acc.append(metric.result().numpy())

        if epoch % 1 == 0:

            metric = keras.metrics.CategoricalAccuracy()
            for x, y in tqdm(test_loader):
                y = tf.one_hot(y, depth=2)

                logits = model.predict(x)

                metric.update_state(y, logits)
            print('train_acc:', train_acc[-1], 'test_acc:', metric.result().numpy())
            print()
            metric.reset_states()

    plt.figure()
    plt.plot(train_loss, '-o')
    plt.plot(train_acc, '-o')
    plt.plot(test_acc, '-o')
    plt.title(str(config.model_fn) + '_' + str(config.units) + '_Loss and Accuracy')
    plt.legend(['train_loss', 'train_acc ' + str(np.max(train_acc)), 'test_acc ' + str(np.max(test_acc))], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('loss / accuracy')
    plt.savefig(str(config.model_fn) + '_' + str(config.units) + '_loss_acc.png')

if __name__ == '__main__':
    config = define_argparser()
    main(config)