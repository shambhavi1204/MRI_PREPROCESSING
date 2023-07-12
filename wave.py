import cv2
import pywt
import numpy as np
import os

# Path to the folders containing ADC and DWI images
adc_folder_path = "D:\\adc\\adc"
dwi_folder_path = "D:\\testing\\dwi"

# Create a new folder for the fused images
output_folder_path = "D:\\adc\\wave"
os.makedirs(output_folder_path, exist_ok=True)

# Fusion method
def fuse_images(coeff_adc, coeff_dwi):
    # Select fusion rule (e.g., maximum, average, etc.)
    fused_coeff = np.maximum(coeff_adc, coeff_dwi)
    return fused_coeff

# Iterate over files in the ADC folder
for adc_filename in os.listdir(adc_folder_path):
    if adc_filename.endswith(".jpg") or adc_filename.endswith(".png"):
        # Read ADC image
        adc_image = cv2.imread(os.path.join(adc_folder_path, adc_filename), cv2.IMREAD_GRAYSCALE)

        # Get corresponding DWI filename
        dwi_filename = adc_filename  # Assuming filenames are the same in both folders

        # Read DWI image
        dwi_image = cv2.imread(os.path.join(dwi_folder_path, dwi_filename), cv2.IMREAD_GRAYSCALE)

        # Invert the DWI image
        inverted_dwi_image = cv2.bitwise_not(dwi_image)

        # Convert pixel values below 1 to 0 and above 0 to 255
        inverted_dwi_image[inverted_dwi_image > 0] = 255

        # Apply wavelet transform on ADC image
        coeffs_adc = pywt.dwt2(adc_image, 'haar')

        # Apply wavelet transform on inverted DWI image
        coeffs_dwi = pywt.dwt2(inverted_dwi_image, 'haar')

        # Fuse the corresponding frequency bands using fusion rules
        fused_coeffs = []
        for (coeff_adc, coeff_dwi) in zip(coeffs_adc, coeffs_dwi):
            fused_coeff = fuse_images(coeff_adc, coeff_dwi)
            fused_coeffs.append(fused_coeff)

        # Reconstruct the fused image from the fused coefficients
        fused_image = pywt.idwt2(fused_coeffs, 'haar')

        # Normalize the fused image if necessary
        fused_image = (fused_image - np.min(fused_image)) / (np.max(fused_image) - np.min(fused_image))
        fused_image = (fused_image * 255).astype(np.uint8)  # Convert back to uint8

        # Create a 3-channel fused image
        fused_image_rgb = cv2.cvtColor(fused_image, cv2.COLOR_GRAY2RGB)

        # Save the fused image in the output folder
        fused_filename = os.path.join(output_folder_path, 'fused_' + adc_filename)
        cv2.imwrite(fused_filename, fused_image_rgb)

        print("Saved:", fused_filename)
