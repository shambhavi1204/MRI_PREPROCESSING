import cv2
import numpy as np
import os

def intensity_based_fusion_average(adc_image, dwi_image):
    # Normalize the ADC and DWI images (optional)
    adc_norm = cv2.normalize(adc_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    dwi_norm = cv2.normalize(dwi_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Perform intensity-based fusion using average fusion
    fused_image = np.zeros_like(adc_image, dtype=np.float32)
    fused_image = (adc_norm.astype(np.float32) + dwi_norm.astype(np.float32)) / 2.0

    # Normalize the fused image back to the original range (optional)
    fused_image = cv2.normalize(fused_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return fused_image


# Path to ADC and DWI image folders
adc_folder = "D:\\adc\\training\\T2"
dwi_folder = "D:\\adc\\training\\dwi"
output_folder = "D:\\adc\\training\\fusion_intensity"

# Get the list of ADC and DWI image files
adc_files = [os.path.join(adc_folder, filename) for filename in os.listdir(adc_folder)]
dwi_files = [os.path.join(dwi_folder, filename) for filename in os.listdir(dwi_folder)]

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over the image files and perform intensity-based fusion
for adc_file, dwi_file in zip(adc_files, dwi_files):
    # Load the ADC and DWI images
    adc_image = cv2.imread(adc_file, cv2.IMREAD_GRAYSCALE)
    dwi_image = cv2.imread(dwi_file, cv2.IMREAD_GRAYSCALE)

    # Perform intensity-based fusion using average fusion
    fused_image = intensity_based_fusion_average(adc_image, dwi_image)

    # Convert the fused image to a 3-channel image
    fused_image_3channel = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2BGR)

    # Create the output filename
    output_filename = os.path.basename(adc_file)  # You can modify this if needed
    output_filepath = os.path.join(output_folder, output_filename)

    # Save the fused image as a 3-channel image
    cv2.imwrite(output_filepath, fused_image_3channel)
