class cnnModel:

    def __init__(self, n_input, n_output):
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten
        from keras.layers import Convolution1D, MaxPooling1D
        from keras.regularizers import l2

        self.n_input = n_input
        self.n_output = n_output

        self.model = Sequential()
        self.model.add(Convolution1D(32, 3, input_shape=(n_input, 1), border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_length=4))
        self.model.add(Convolution1D(32, 3, border_mode='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling1D(pool_length=4))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, W_regularizer=l2(0.01)))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(n_output, W_regularizer=l2(0.01)))
        self.model.add(Activation('softmax'))
        self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    def _format_data(self, data):
        from keras.utils.np_utils import to_categorical

        X = data[0].reshape(data[0].shape + (1,))
        y = to_categorical(data[1], self.n_output)
        return X, y

    def train_model(self, train_data, test_data, batch_size=16, nb_epoch=5):
        formated_train = self._format_data(train_data)
        formated_test = self._format_data(test_data)
        self.model.fit(formated_train[0], formated_train[1],
                       validation_split=0., validation_data=formated_test,
                       batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)
        self.model.save_weights("weights.h5", overwrite=True)

    def get_layer_output(self, i_layer, inputs):
        from keras import backend as K

        self.model.load_weights("weights.h5")

        layers = self.model.layers
        get_output = K.function([layers[0].input], [layers[i_layer].output])
        return get_output([inputs.reshape(inputs.shape + (1,))])[0]


if __name__ == '__main__':
    import cwru
    import numpy as np
    from scipy.fftpack import rfft
    import random
    from utils import gplot_images, gplot_lines
    cwru_data = cwru.CWRU("12DriveEndFault", "1797", 384)
    X_train, X_test = cwru_data.X_train, cwru_data.X_test
    X_train_rfft, X_test_rfft = rfft(X_train), rfft(X_test)
    y_train, y_test = cwru_data.y_train, cwru_data.y_test

    cnn_model = cnnModel(cwru_data.length, cwru_data.nclasses)
    cnn_model.train_model((X_train_rfft, y_train), (X_test_rfft, y_test))

    from collections import defaultdict
    indices = defaultdict(list)

    # group y_test
    for i, c in enumerate(y_test):
        indices[c].append(i)

    # 3 random samples for each class
    inds = sum([random.sample(indices[c], 3) for c in indices], [])
    inputs = X_test[inds]
    inputs_rfft = X_test_rfft[inds]

    # short titles
    short_titles = [title.replace('-Ball', 'b')
                    .replace('-OuterRace3', 'o3')
                    .replace('-OuterRace6', 'o6')
                    .replace('-OuterRace12', 'o12')
                    .replace('-InnerRace', 'in')
                    .replace('0.0', '')
                    .replace('Normal', 'no')
                    for title in cwru_data.labels]
    gplot_lines(inputs, short_titles, 'images/signals.png')
    gplot_lines(inputs_rfft, short_titles, 'images/freqs.png')

    layer_output = cnn_model.get_layer_output(5, inputs_rfft)
    gplot_images(layer_output, short_titles, 'images/pooling.png')

    layer_output_flatten = layer_output.reshape((layer_output.shape[0], -1))
    gplot_lines(layer_output_flatten, short_titles, 'images/pooing_flatten.T.png')
