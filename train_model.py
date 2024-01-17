import tensorflow as tf
import matplotlib.pyplot as plt
from model_definition import unet_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load the dataset
# X and Y are assumed to be pre-loaded arrays of training images and masks
X = np.load('path_to_training_images.npy')
Y = np.load('path_to_training_masks.npy')

# Split data into train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1)

# Data augmentation generators
train_image_data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)
train_mask_data_generator = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Seed for reproducibility
seed = 1

# Define the train and validation generators
train_generator = zip(
    train_image_data_generator.flow(X_train, batch_size=16, seed=seed),
    train_mask_data_generator.flow(Y_train, batch_size=16, seed=seed)
)
val_generator = zip(
    train_image_data_generator.flow(X_val, batch_size=16, seed=seed),
    train_mask_data_generator.flow(Y_val, batch_size=16, seed=seed)
)

# Initialize U-Net model
input_size = X_train.shape[1:]  # Assumes X_train is of shape (n_samples, height, width, channels)
model = unet_model(input_size)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 16,
    epochs=50,
    validation_data=val_generator,
    validation_steps=len(X_val) // 16
)

# Save model
model.save('models/final_model.h5')

# Optionally, plot the training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.legend()

plt.show()
