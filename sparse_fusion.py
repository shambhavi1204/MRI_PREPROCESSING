import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
import cv2

def sparse_fusion(adc_image, dwi_image):
    
    adc_vector = adc_image.reshape(-1, 1)
    dwi_vector = dwi_image.reshape(-1, 1)

    
    input_data = np.concatenate((adc_vector, dwi_vector), axis=1)

    # Perform sparse coding using Orthogonal Matching Pursuit
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)  # Adjust the number of nonzero coefficients as needed
    omp.fit(input_data, adc_vector)

    # Obtain the sparse coefficients
    adc_coeffs = omp.coef_
    dwi_coeffs = np.sum(omp.transform(input_data) - adc_coeffs, axis=1)

    # Combine the coefficients
    fused_coeffs = adc_coeffs + dwi_coeffs

    # Reconstruct the fused image
    fused_image = np.dot(input_data, fused_coeffs)

    
    fused_image = fused_image.reshape(adc_image.shape)

    return fused_image

# Example usage
adc_image = cv2.imread(r"D:\adc\adc\01_028.jpg", cv2.IMREAD_GRAYSCALE)
dwi_image = cv2.imread(r"D:\testing\dwi\01_028.jpg", cv2.IMREAD_GRAYSCALE)
fused_image = sparse_fusion(adc_image, dwi_image)
print(fused_image)
