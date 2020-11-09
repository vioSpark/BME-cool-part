import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test_image.jpg', -1)
print(img.shape)
R, G, B = (img[:, :, 0], img[:, :, 1], img[:, :, 2])
dftR = cv2.dft(np.float32(R), flags=cv2.DFT_COMPLEX_OUTPUT)
dftG = cv2.dft(np.float32(G), flags=cv2.DFT_COMPLEX_OUTPUT)
dftB = cv2.dft(np.float32(B), flags=cv2.DFT_COMPLEX_OUTPUT)
"""dft_shiftR = np.fft.fftshift(dftR)
dft_shiftG = np.fft.fftshift(dftG)
dft_shiftB = np.fft.fftshift(dftB)"""

dft_shiftR = dftR
dft_shiftG = dftG
dft_shiftB = dftB

magnitude_spectrumR = 20 * np.log(cv2.magnitude(dft_shiftR[:, :, 0], dft_shiftR[:, :, 1]))
magnitude_spectrumG = 20 * np.log(cv2.magnitude(dft_shiftG[:, :, 0], dft_shiftG[:, :, 1]))
magnitude_spectrumB = 20 * np.log(cv2.magnitude(dft_shiftB[:, :, 0], dft_shiftB[:, :, 1]))

shape = list(magnitude_spectrumR.shape)
shape.append(1)
shape = tuple(shape)
magnitude_spectrumRt = np.reshape(magnitude_spectrumR, shape)
magnitude_spectrumGt = np.reshape(magnitude_spectrumG, shape)
magnitude_spectrumBt = np.reshape(magnitude_spectrumB, shape)

RGB_magnitude = np.concatenate((magnitude_spectrumRt, magnitude_spectrumGt, magnitude_spectrumBt), axis=2)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(RGB_magnitude.astype('uint8'), cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

"""
rows, cols, channels = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 200:crow + 200, ccol - 400:ccol + 400] = 1

# apply mask and inverse DFT
fshiftR = dft_shiftR * mask
fshiftG = dft_shiftG * mask
fshiftB = dft_shiftB * mask
"""
"""
f_ishiftR = np.fft.ifftshift(fshiftR)
f_ishiftG = np.fft.ifftshift(fshiftG)
f_ishiftB = np.fft.ifftshift(fshiftB)
"""
f_ishiftR = np.fft.ifftshift(dft_shiftR)
f_ishiftG = np.fft.ifftshift(dft_shiftG)
f_ishiftB = np.fft.ifftshift(dft_shiftB)

img_backR = cv2.idft(dftR)
img_backG = cv2.idft(dftG)
img_backB = cv2.idft(dftB)

img_backR = cv2.magnitude(img_backR[:, :, 0], img_backR[:, :, 1])
img_backG = cv2.magnitude(img_backG[:, :, 0], img_backG[:, :, 1])
img_backB = cv2.magnitude(img_backB[:, :, 0], img_backB[:, :, 1])

# building back RGB
img_backR = np.reshape(img_backR, shape)
img_backG = np.reshape(img_backG, shape)
img_backB = np.reshape(img_backB, shape)

img_back = np.concatenate((img_backR, img_backG, img_backB), axis=2)

plt.subplot(223), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_back.astype('uint8'), cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
