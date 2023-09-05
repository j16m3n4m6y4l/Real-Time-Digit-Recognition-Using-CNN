import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

# Main Hyper-parameters
batch_size = 256
num_classes = 10
epochs = 12

# Get dataset
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Specify samples used for training and testing
train_samples = 60000
test_samples = 10000

x_train = x_train[:train_samples, :]
y_train = y_train[:train_samples]
x_test = x_test[:test_samples, :]
y_test = y_test[:test_samples]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize into 0 - 1
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu',
                     input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer='adam',
                  metrics=['accuracy'])

# Fit data to model
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1] * 100)

# Prediction
y_predicted = model.predict(x_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_rounded_labels = np.argmax(y_test, axis=1)

model.summary()

model.save('model.h5')