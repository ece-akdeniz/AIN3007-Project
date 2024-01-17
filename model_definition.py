import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, 
                                     Concatenate, Conv2DTranspose)
from tensorflow.keras.models import Model

def conv_block(input_tensor, num_filters):
    encoder = Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.activations.relu(encoder)
    encoder = Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = tf.keras.layers.BatchNormalization()(encoder)
    encoder = tf.keras.activations.relu(encoder)
    return encoder

def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder

def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = Concatenate(axis=-1)([decoder, concat_tensor])
    decoder = conv_block(decoder, num_filters)
    return decoder

def unet_model(input_size):
    inputs = Input(input_size)

    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)

    center = conv_block(encoder4_pool, 1024)

    decoder4 = decoder_block(center, encoder4, 512)
    decoder3 = decoder_block(decoder4, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(decoder0)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Example usage:
# Define input size according to your data, e.g., (256, 256, 3) for 256x256 RGB images
input_size = (256, 256, 3)
model = unet_model(input_size)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save the model architecture to a JSON file
with open('unet_model_architecture.json', 'w') as f:
    f.write(model.to_json())
