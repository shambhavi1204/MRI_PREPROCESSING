import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Folder paths
adc_folder = "D:\\adc\\adc"
dwi_folder = "D:\\testing\\dwi"
fusion_folder = "D:\\testing\\is"
output_folder = "D:\\testing\\merged_images11"

# Get the list of ADC, DWI, and fused image files
adc_files = [os.path.join(adc_folder, filename) for filename in os.listdir(adc_folder)]
dwi_files = [os.path.join(dwi_folder, filename) for filename in os.listdir(dwi_folder)]
fusion_files = [os.path.join(fusion_folder, filename) for filename in os.listdir(fusion_folder)]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over the image files and merge them
for adc_file, dwi_file, fusion_file in zip(adc_files, dwi_files, fusion_files):
    # Load the ADC, DWI, and fused images using plt.imread()
    adc_image = plt.imread(adc_file)
    dwi_image = plt.imread(dwi_file)
    fused_image = plt.imread(fusion_file)
    fused_image = np.expand_dims(fused_image, axis=-1)  # Add a channel dimension
    fused_image = np.repeat(fused_image, 3, axis=-1) 

    # Calculate the mean of each image
    adc_mean = np.mean(adc_image, axis=-1, keepdims=True)
    dwi_mean = np.mean(dwi_image, axis=-1, keepdims=True)
    fused_mean = np.mean(fused_image, axis=-1, keepdims=True)

    # Normalize the mean images using cv2.normalize()
    adc_norm = cv2.normalize(adc_mean, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dwi_norm = cv2.normalize(dwi_mean, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    fused_norm = cv2.normalize(fused_mean, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Merge the three normalized images into a single image
    merged_image = cv2.merge((adc_norm, fused_norm, dwi_norm))

    # Create the output filename
    filename = os.path.basename(adc_file)  # You can modify this if needed
    output_file = os.path.join(output_folder, filename)
    print(filename)

    # Save the merged image
    cv2.imwrite(output_file, merged_image)
