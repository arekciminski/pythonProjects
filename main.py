from liby_clasyfiaction import Dataset, DataGenerator, prepare_data
# define model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import numpy as np

classification_dataset = Dataset()

print('---------------------------------')
print('Poczatek tworzenia zbioru treningowego')

NUMBER_OF_TRAIN_IMAGES = 35000
NUMBER_OF_VAL_IMAGES = 5000
scale_percent = 60

x_train,y_train, x_val,y_val,x_size,y_size = prepare_data(NUMBER_OF_TRAIN_IMAGES,NUMBER_OF_VAL_IMAGES,scale_percent)

y_train = np.array(y_train)
y_train = np.where(y_train == -1,0,y_train)

print('Koniec tworzenia zbioru walidacyjnego')
print('---------------------------------')

train_data_generator = DataGenerator(
    x_train, y_train, batch_size=64, shuffle=False, augment=False
)

model = Sequential()
model.add(Flatten(input_shape=(x_size, y_size, 3)))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='tanh'))

# compile model

print('---------------------------------')
print('Poczatek uczenia sieci')

# fit model
# compile model
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
# fit model
history = model.fit(train_data_generator, epochs=10, validation_data=(np.array(x_val), np.array(y_val)), verbose='auto')
