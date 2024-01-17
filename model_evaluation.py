import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import matplotlib.pyplot as plt

def load_test_data():
    # Load your test data here
    # X_test: Test images
    # Y_test: Corresponding ground truth masks
    X_test = np.load('path_to_test_images.npy')
    Y_test = np.load('path_to_test_masks.npy')
    return X_test, Y_test

def evaluate_model(model_path, X_test, Y_test):
    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model on test data
    results = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Loss: {results[0]}")
    print(f"Test Accuracy: {results[1]}")

    # Optionally, calculate additional metrics like F1-score
    # Y_pred = model.predict(X_test)
    # f1 = f1_score(Y_test.flatten(), Y_pred.flatten() > 0.5)
    # print(f"Test F1 Score: {f1}")

def visualize_predictions(model_path, X_test, num_samples=5):
    model = load_model(model_path)
    predictions = model.predict(X_test[:num_samples])

    fig, ax = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))
    for i in range(num_samples):
        ax[i, 0].imshow(X_test[i])
        ax[i, 0].set_title("Original Image")

        ax[i, 1].imshow(predictions[i].squeeze(), cmap='gray')
        ax[i, 1].set_title("Predicted Mask")

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    model_path = 'path_to_your_trained_model.h5'
    X_test, Y_test = load_test_data()
    evaluate_model(model_path, X_test, Y_test)
    visualize_predictions(model_path, X_test)
