import SimpleITK as sitk
import numpy as np

# Load DWI and T2 images
dwi_image = sitk.ReadImage("D:\\SISS2015_Training\\1\\VSD.Brain.XX.O.MR_DWI.70613\VSD.Brain.XX.O.MR_DWI.70613.nii")
t2_image = sitk.ReadImage("D:\\SISS2015_Training\\1\\VSD.Brain.XX.O.MR_T2.70616\VSD.Brain.XX.O.MR_T2.70616.nii")

# Perform intensity normalization
dwi_array = sitk.GetArrayFromImage(dwi_image)
t2_array = sitk.GetArrayFromImage(t2_image)

dwi_mean = np.mean(dwi_array)
dwi_std = np.std(dwi_array)
t2_mean = np.mean(t2_array)
t2_std = np.std(t2_array)

normalized_dwi_array = (dwi_array - dwi_mean) / dwi_std
normalized_t2_array = (t2_array - t2_mean) / t2_std

normalized_dwi_image = sitk.GetImageFromArray(normalized_dwi_array)
normalized_dwi_image.CopyInformation(dwi_image)
normalized_t2_image = sitk.GetImageFromArray(normalized_t2_array)
normalized_t2_image.CopyInformation(t2_image)


registration_method = sitk.ImageRegistrationMethod()


registration_method.SetMetricAsMeanSquares()


registration_method.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=100)

# Set the interpolator to linear
registration_method.SetInterpolator(sitk.sitkLinear)


initial_transform = sitk.CenteredTransformInitializer(normalized_dwi_image, normalized_t2_image, sitk.AffineTransform(dwi_image.GetDimension()))
registration_method.SetInitialTransform(initial_transform)

# Perform the registration
final_transform = registration_method.Execute(normalized_dwi_image, normalized_t2_image)


registered_normalized_dwi_image = sitk.Resample(normalized_dwi_image, normalized_t2_image, final_transform, sitk.sitkLinear, 0.0, normalized_dwi_image.GetPixelID())

# Denormalize the registered DWI image
registered_dwi_array = (sitk.GetArrayFromImage(registered_normalized_dwi_image) * dwi_std) + dwi_mean
registered_dwi_image = sitk.GetImageFromArray(registered_dwi_array)
registered_dwi_image.CopyInformation(dwi_image)

sitk.WriteImage(registered_dwi_image, "D:\\SISS2015_Training\\1\\fused1.nii")
