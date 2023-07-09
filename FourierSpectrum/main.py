
#Yasemin KOÇ 190316035
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
from scipy.ndimage import median_filter

# Load the noisy input image
im1 = plt.imread('image63.tif')

# Apply median filter to im1
im2 = median_filter(im1, size=0.3)

# Compute the shifted Fourier spectrum of im1
im4 = fftpack.fftshift(fftpack.fft2(im1))

# Define the noise points and notch filter radius
noise_points = [(114, 114), (124, 122), (132, 134), (142, 142)]
notch_radius = 5

# Create the custom notch filter
im5 = np.ones(im4.shape)
for point in noise_points:
    x, y = point
    # Generate coordinates grid
    y_coords, x_coords = np.ogrid[:im4.shape[0], :im4.shape[1]]
    # Calculate distance from each coordinate to the noise point
    distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
    # Create Butterworth filter mask
    mask = 1 - np.exp(-distances**2 / (2 * notch_radius**2))
    im5 *= mask

# Apply the notch filter to the Fourier spectrum of im1
im6 = im4 * im5
im3 = np.abs(fftpack.ifft2(fftpack.ifftshift(im6)))

# Plotting the figures
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].imshow(im1, cmap='gray')
axes[0, 0].set_title('Noisy Image')

axes[0, 1].imshow(im2, cmap='gray')
axes[0, 1].set_title('Median Filtered Image')

axes[0, 2].imshow(im3, cmap='gray')
axes[0, 2].set_title('Notch Filtered Image')

axes[1, 0].imshow(np.log(1 + np.abs(im4)), cmap='gray')
axes[1, 0].set_title('Fourier Spectrum of Noisy Image')

axes[1, 1].imshow(np.log(1 + np.abs(im5)), cmap='gray')
axes[1, 1].set_title('Notch Filters')

axes[1, 2].imshow(np.log(1 + np.abs(im6)), cmap='gray')
axes[1, 2].set_title('Fourier Spectrum of Filtered Image')

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()

