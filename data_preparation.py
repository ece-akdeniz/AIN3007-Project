import openslide
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from skimage.transform import resize
import numpy as np
import cv2
import os
from pathlib import Path
import matplotlib.pyplot as plt
import imageio
import argparse

def preprocess_thumbnail_for_model(thumbnail, target_size=(256, 256)):
    # Resize thumbnail to expected input size of the model
    thumbnail_resized = resize(thumbnail, target_size, preserve_range=True)

    # Use a pre-processing function suitable for your model architecture
    # For instance, if using a ResNet-based U-Net model, you might do:
    thumbnail_preprocessed = preprocess_input(thumbnail_resized)

    # Expand dimensions to add batch size of 1 at the start
    thumbnail_preprocessed = np.expand_dims(thumbnail_preprocessed, axis=0)

    return thumbnail_preprocessed


def postprocess_predicted_mask(mask, threshold=0.5):
    # Apply threshold to binary predictions
    mask_thresholded = (mask > threshold).astype(np.uint8)

    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_thresholded, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask_cleaned


def generate_tissue_mask(slide_path, model, level=6, target_size=(256, 256)):
    slide = openslide.OpenSlide(slide_path)
    thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
    thumbnail = np.array(thumbnail)[:,:,:3]

    # Preprocess the thumbnail for the model
    thumbnail_preprocessed = preprocess_thumbnail_for_model(thumbnail, target_size)

    # Predict the mask using the model
    predicted_mask = model.predict(thumbnail_preprocessed)

    # Post-process the predicted mask
    processed_mask = postprocess_predicted_mask(predicted_mask.squeeze())

    return processed_mask, slide


# Define a function to crop patches from the WSI based on the tissue mask
def crop_patches_from_mask(slide, mask, level=1, patch_size=512, stride=512):
    downsample_factor = slide.level_downsamples[level]
    num_patches = 0
    
    mask = cv2.resize(mask, slide.level_dimensions[level])
    mask = (mask > 127).astype(np.uint8) * 255  # Ensure mask is binary

    # Calculate the number of patches to extract
    for y in range(0, mask.shape[0], stride):
        for x in range(0, mask.shape[1], stride):
            mask_patch = mask[y:y+patch_size, x:x+patch_size]
            if mask_patch.sum() > 0:  # If there's tissue in the patch
                patch = slide.read_region((int(x * downsample_factor), int(y * downsample_factor)), level, (patch_size, patch_size))
                patch = np.array(patch)[:,:,:3]
                yield patch  # This will return a generator of patches
                num_patches += 1
