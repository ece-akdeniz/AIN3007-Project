import tensorflow as tf
from tensorflow.keras.models import load_model

def convert_to_tflite(model_path, tflite_model_path):
    # Load the Keras model
    model = load_model(model_path)

    # Convert the model to the TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to disk
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted and saved to {tflite_model_path}")

def main():
    model_path = 'path_to_your_trained_model.h5'
    tflite_model_path = 'path_to_save_tflite_model.tflite'
    convert_to_tflite(model_path, tflite_model_path)

if __name__ == '__main__':
    main()
