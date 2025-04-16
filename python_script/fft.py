import numpy as np
from skimage import io, color
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import matplotlib.pyplot as plt

# 读取图片并转换为灰度图
img = io.imread(r'..\figures\ustcblack.jpg')
gray_img = color.rgb2gray(img)

# 进行傅里叶变换
fourier = fft2(gray_img)
shifted_fourier = fftshift(fourier)
magnitude_spectrum = np.log(np.abs(shifted_fourier) + 1e-10)

# 保存傅里叶变换后的幅度谱
plt.imsave('fourier_transform_skimage.jpg', magnitude_spectrum, cmap='gray')

# 滤掉高频部分
rows, cols = gray_img.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols))
radius = 50
mask[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1
filtered_fourier = shifted_fourier * mask

# 保存滤波后的幅度谱
filtered_spectrum = np.log(np.abs(filtered_fourier) + 1e-10)
plt.imsave('filtered_spectrum_skimage.jpg', filtered_spectrum, cmap='gray')

# 进行逆傅里叶变换
ishifted_fourier = ifftshift(filtered_fourier)
reconstructed_img = np.abs(ifft2(ishifted_fourier))

# 保存逆傅里叶变换后的图片
plt.imsave('inverse_fourier_transform_skimage.jpg', reconstructed_img, cmap='gray')

# 保存原始图片
io.imsave('original_image_skimage.jpg', (gray_img * 255).astype(np.uint8))
