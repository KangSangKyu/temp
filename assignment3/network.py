from tensorflow.keras import layers, models

class RNN_FC(models.Model):

  def __init__(self, num_classes, max_features, max_len, units):
    super(RNN_FC, self).__init__()

    self.num_classes = num_classes 
    self.max_features = max_features
    self.max_len = max_len  
    self.units = units
    self.embedding = layers.Embedding(self.max_features, 128)
    self.rnn = layers.SimpleRNN(self.units)
    self.fc = layers.Dense(self.num_classes, activation='sigmoid')

  def call(self, x):
    x = self.embedding(x)
    x = self.rnn(x)
    x = self.fc(x)
    return x


class LSTM_FC(models.Model):

    def __init__(self, num_classes, max_features, max_len, units):
        super(LSTM_FC, self).__init__()

        self.num_classes = num_classes
        self.max_features = max_features
        self.max_len = max_len
        self.units = units
        self.embedding = layers.Embedding(self.max_features, 128)
        self.lstm = layers.LSTM(self.units)
        self.fc = layers.Dense(self.num_classes, activation="sigmoid")

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm(x)
        x = self.fc(x)
        return x


class TWO_STACKED_LSTM_FC(models.Model):

    def __init__(self, num_classes, max_features, max_len, units):
        super(TWO_STACKED_LSTM_FC, self).__init__()

        self.num_classes = num_classes
        self.max_features = max_features
        self.max_len = max_len
        self.units = units
        self.embedding = layers.Embedding(self.max_features, 128)
        self.lstm_1 = layers.LSTM(self.units, return_sequences=True)
        self.lstm_2 = layers.LSTM(self.units, return_sequences=False)
        self.fc = layers.Dense(self.num_classes, activation="sigmoid")

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.fc(x)
        return x


class THREE_STACKED_LSTM_FC(models.Model):

    def __init__(self, num_classes, max_features, max_len, units):
        super(THREE_STACKED_LSTM_FC, self).__init__()

        self.num_classes = num_classes
        self.max_features = max_features
        self.max_len = max_len
        self.units = units
        self.embedding = layers.Embedding(self.max_features, 128)
        self.lstm_1 = layers.LSTM(self.units, return_sequences=True)
        self.lstm_2 = layers.LSTM(self.units, return_sequences=True)
        self.lstm_3 = layers.LSTM(self.units, return_sequences=False)
        self.fc = layers.Dense(self.num_classes, activation="sigmoid")

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = self.fc(x)
        return x


class FOUR_STACKED_LSTM_FC(models.Model):

    def __init__(self, num_classes, max_features, max_len, units):
        super(FOUR_STACKED_LSTM_FC, self).__init__()

        self.num_classes = num_classes
        self.max_features = max_features
        self.max_len = max_len
        self.units = units
        self.embedding = layers.Embedding(self.max_features, 128)
        self.lstm_1 = layers.LSTM(self.units, return_sequences=True)
        self.lstm_2 = layers.LSTM(self.units, return_sequences=True)
        self.lstm_3 = layers.LSTM(self.units, return_sequences=True)
        self.lstm_4 = layers.LSTM(self.units, return_sequences=False)
        self.fc = layers.Dense(self.num_classes, activation="sigmoid")

    def call(self, x):
        x = self.embedding(x)
        x = self.lstm_1(x)
        x = self.lstm_2(x)
        x = self.lstm_3(x)
        x = self.lstm_4(x)
        x = self.fc(x)
        return x