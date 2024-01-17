import argparse
from data_preparation import generate_tissue_mask, crop_patches_from_mask
from model_definition import unet_model
from tensorflow.keras.models import load_model
import os
import numpy as np
import imageio

def main(args):
    # Load the U-Net model
    model = load_model('unet.h5')

    # Read the list of WSI paths
    with open(args.wsi_paths, 'r') as file:
        wsi_paths = file.readlines()

        # Process each WSI
        for wsi_path in wsi_paths:
            mask, slide = generate_tissue_mask(wsi_path)
            patch_generator = crop_patches_from_mask(slide, mask)
            
            # Save the tissue mask to a file
            for i, patch in enumerate(patch_generator):
                patch_filename = os.path.join(output_folder, f"{Patch(wsi_path).stem}_patch_{i}.png")
                imageio.imwrite(patch_filename, patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment tissue regions from WSIs.')
    parser.add_argument('--wsi_paths', type=str, help='File containing the list of paths to WSIs')
    parser.add_argument('--output_folder', type=str, default='./predicted_tissue_masks', help='Folder to save tissue masks')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    main(args)
