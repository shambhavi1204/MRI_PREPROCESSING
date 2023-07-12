import cv2
import numpy as np
import os


dwi_folder_path = "D:\\testing\\dwi"

# Create a new folder for the inverted images
inverted_folder_path = "D:\\testing\\inverted_folder"
os.makedirs(inverted_folder_path, exist_ok=True)


for filename in os.listdir(dwi_folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Read DWI image
        dwi_image = cv2.imread(os.path.join(dwi_folder_path, filename), cv2.IMREAD_GRAYSCALE)

        
        _, thresholded = cv2.threshold(dwi_image, 1, 255, cv2.THRESH_BINARY_INV)

       
        inverted_image = cv2.bitwise_not(thresholded)

        # Combine the inverted foreground with the background
        inverted_dwi_image = cv2.bitwise_or(dwi_image, inverted_image)

        # Save the inverted DWI image in the output folder
        output_filename = os.path.join(inverted_folder_path, filename)
        cv2.imwrite(output_filename, inverted_dwi_image)

        print("Saved:", output_filename)
