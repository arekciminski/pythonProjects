from liby_clasyfiaction import Dataset, DataGenerator,reshape_image

import cv2
import tensorflow as tf
import keras
import numpy as np
#from pandas_profiling import ProfileReport

classification_dataset = Dataset()

print('---------------------------------')
print('Poczatek tworzenia zbioru treningowego')

NUMBER_OF_TRAIN_IMAGES = 35000

x_train = []
y_train = []
i = 0
scale_percent = 60
for (index, image), label in zip(classification_dataset.df.iterrows(), classification_dataset.df[['image_id','Male']].values):
  if image['partition'] == 0:
    i = i +1
    resized_image = reshape_image(cv2.imread(classification_dataset.images_path + image['image_id']), scale_percent)
    x_train.append(resized_image)
    y_train.append(label[1])
  if i == NUMBER_OF_TRAIN_IMAGES:
    break

x_size = resized_image.shape[0]
y_size = resized_image.shape[1]
print('Koniec tworzenia zbioru treningowego')
print('---------------------------------')
print('Poczatek tworzenia zbioru walidacyjnego')


NUMBER_OF_VAL_IMAGES = 5000

x_val = []
y_val = []
i = NUMBER_OF_TRAIN_IMAGES
for (index, image), label in zip(classification_dataset.df.iterrows(), classification_dataset.df[['image_id','Male']].values):
  if image['partition'] == 1:
    i = i +1
    resized_image = reshape_image(cv2.imread(classification_dataset.images_path + image['image_id']), scale_percent)
    x_val.append(resized_image)
    y_val.append(label[1])
  if i == NUMBER_OF_TRAIN_IMAGES+NUMBER_OF_VAL_IMAGES:
    break

print('Koniec tworzenia zbioru walidacyjnego')
print('---------------------------------')

train_data_generator = DataGenerator(
    x_train, y_train, batch_size=64, shuffle=False, augment=False
)

# define model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(x_size, y_size, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='tanh'))

# compile model
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print('---------------------------------')
print('Poczatek uczenia sieci')

# fit model
history = model.fit(np.array(x_train), np.array(y_train), epochs=30, batch_size=16, validation_data=(np.array(x_val), np.array(y_val)), verbose=1)

ProfileReport(history)