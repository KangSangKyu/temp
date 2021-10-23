import  tensorflow as tf
from    tensorflow import  keras
from    tensorflow.keras import datasets, layers, optimizers, models
from    tensorflow.keras import regularizers

class VGG16_A(models.Model):

  def __init__(self, dropout_ratio=.0):

    super(VGG16_A, self).__init__()

    self.num_classes = 10
    self.dropout_ratio = dropout_ratio

    self.conv_block1 = keras.Sequential([
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block2 = keras.Sequential([
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block3 = keras.Sequential([
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block4 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block5 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.flatten = layers.Flatten()
    self.dropout1 = layers.Dropout(dropout_ratio)
    self.fc1 = layers.Dense(512)
    self.dropout2 = layers.Dropout(dropout_ratio)
    self.fc2 = layers.Dense(512)
    self. dropout3 = layers.Dropout(dropout_ratio)
    self.classification = layers.Dense(self.num_classes, activation='softmax')

  def call(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = self.conv_block5(x)
    x = self.flatten(x)
    x = self.dropout1(x)
    x = self.fc1(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    x = self.dropout3(x)
    x = self.classification(x)
    return x


class VGG16_B(models.Model):

  def __init__(self, dropout_ratio=.0):

    super(VGG16_B, self).__init__()

    self.num_classes = 10
    self.dropout_ratio = dropout_ratio

    self.conv_block1 = keras.Sequential([
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block2 = keras.Sequential([
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block3 = keras.Sequential([
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block4 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block5 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.flatten = layers.Flatten()
    self.dropout1 = layers.Dropout(dropout_ratio)
    self.fc1 = layers.Dense(512)
    self.dropout2 = layers.Dropout(dropout_ratio)
    self.fc2 = layers.Dense(512)
    self.dropout3 = layers.Dropout(dropout_ratio)
    self.classification = layers.Dense(self.num_classes, activation='softmax')

  def call(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = self.conv_block5(x)
    x = self.flatten(x)
    x = self.dropout1(x)
    x = self.fc1(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    x = self.dropout3(x)
    x = self.classification(x)
    return x

class VGG16_C(models.Model):

  def __init__(self, dropout_ratio=.0):

    super(VGG16_C, self).__init__()

    self.num_classes = 10
    self.dropout_ratio = dropout_ratio

    self.conv_block1 = keras.Sequential([
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block2 = keras.Sequential([
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block3 = keras.Sequential([
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block4 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block5 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (1, 1), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.flatten = layers.Flatten()
    self.dropout1 = layers.Dropout(dropout_ratio)
    self.fc1 = layers.Dense(512)
    self.dropout2 = layers.Dropout(dropout_ratio)
    self.fc2 = layers.Dense(512)
    self.dropout3 = layers.Dropout(dropout_ratio)
    self.classification = layers.Dense(self.num_classes, activation='softmax')

  def call(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = self.conv_block5(x)
    x = self.flatten(x)
    x = self.dropout1(x)
    x = self.fc1(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    x = self.dropout3(x)
    x = self.classification(x)
    return x


class VGG16_D(models.Model):

    def __init__(self, dropout_ratio=.0):
        """
        :param input_shape: [32, 32, 3]
        """
        super(VGG16_D, self).__init__()

        self.num_classes = 10
        self.dropout_ratio = dropout_ratio

        self.conv_block1 = keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block2 = keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block3 = keras.Sequential([
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block4 = keras.Sequential([
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block5 = keras.Sequential([
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(dropout_ratio)
        self.fc1 = layers.Dense(512)
        self.dropout2 = layers.Dropout(dropout_ratio)
        self.fc2 = layers.Dense(512)
        self.dropout3 = layers.Dropout(dropout_ratio)
        self.classification = layers.Dense(self.num_classes, activation='softmax')

    def call(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.classification(x)
        return x

class VGG16_E(models.Model):

    def __init__(self, dropout_ratio=.0):
        """
        :param input_shape: [32, 32, 3]
        """
        super(VGG16_E, self).__init__()

        self.num_classes = 10
        self.dropout_ratio = dropout_ratio

        self.conv_block1 = keras.Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block2 = keras.Sequential([
            layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)   
        ])
        self.conv_block3 = keras.Sequential([
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block4 = keras.Sequential([
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.conv_block5 = keras.Sequential([
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(dropout_ratio)
        ])
        self.flatten = layers.Flatten()
        self.dropout1 = layers.Dropout(dropout_ratio)
        self.fc1 = layers.Dense(512)
        self.dropout2 = layers.Dropout(dropout_ratio)
        self.fc2 = layers.Dense(512)
        self.dropout3 = layers.Dropout(dropout_ratio)
        self.classification = layers.Dense(self.num_classes, activation='softmax')

    def call(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.conv_block5(x)
        x = self.flatten(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.classification(x)
        return x

class VGG16_F(models.Model):

  def __init__(self, dropout_ratio=.0):

    super(VGG16_F, self).__init__()

    self.num_classes = 10
    self.dropout_ratio = dropout_ratio

    self.conv_block1 = keras.Sequential([
      layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block2 = keras.Sequential([
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block3 = keras.Sequential([
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block4 = keras.Sequential([
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block5 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(1, 1)),
      layers.Dropout(dropout_ratio)
    ])
    self.flatten = layers.Flatten()
    self.dropout1 = layers.Dropout(dropout_ratio)
    self.fc1 = layers.Dense(512)
    self.dropout2 = layers.Dropout(dropout_ratio)
    self.fc2 = layers.Dense(512)
    self.dropout3 = layers.Dropout(dropout_ratio)
    self.classification = layers.Dense(self.num_classes, activation='softmax')

  def call(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = self.conv_block5(x)
    x = self.flatten(x)
    x = self.dropout1(x)
    x = self.fc1(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    x = self.dropout3(x)
    x = self.classification(x)
    return x

class VGG16_G(models.Model):

  def __init__(self, dropout_ratio=.0):

    super(VGG16_G, self).__init__()

    self.num_classes = 10
    self.dropout_ratio = dropout_ratio

    self.conv_block1 = keras.Sequential([
      layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block2 = keras.Sequential([
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block3 = keras.Sequential([
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block4 = keras.Sequential([
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block5 = keras.Sequential([
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(1, 1)),
      layers.Dropout(dropout_ratio)
    ])
    self.conv_block6 = keras.Sequential([
      layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', activation='relu'),
      layers.MaxPooling2D(pool_size=(2, 2)),
      layers.Dropout(dropout_ratio)
    ])
    self.flatten = layers.Flatten()
    self.dropout1 = layers.Dropout(dropout_ratio)
    self.fc1 = layers.Dense(512)
    self.dropout2 = layers.Dropout(dropout_ratio)
    self.fc2 = layers.Dense(512)
    self.dropout3 = layers.Dropout(dropout_ratio)
    self.classification = layers.Dense(self.num_classes, activation='softmax')

  def call(self, x):
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)
    x = self.conv_block4(x)
    x = self.conv_block5(x)
    x = self.conv_block6(x)
    x = self.flatten(x)
    x = self.dropout1(x)
    x = self.fc1(x)
    x = self.dropout2(x)
    x = self.fc2(x)
    x = self.dropout3(x)
    x = self.classification(x)
    return x