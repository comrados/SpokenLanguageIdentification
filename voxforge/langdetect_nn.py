import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_train = r"D:\speechrecogn\voxforge\pics\train"
path_valid = r"D:\speechrecogn\voxforge\pics\validation"

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        path_train,
        target_size=(150, 513),
        batch_size=32)

validation_generator = validation_datagen.flow_from_directory(
        path_valid,
        target_size=(150, 513),
        batch_size=32)

path = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"

cb = tf.keras.callbacks.ModelCheckpoint(path, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150*513 with 1 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 513, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

from tensorflow.keras.optimizers import RMSprop

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

history = model.fit_generator(train_generator, epochs=2, validation_data = validation_generator, verbose = 1, callbacks=[cb])