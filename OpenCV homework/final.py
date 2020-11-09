import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('test_image.jpg', -1)
print(img.shape)

dft = [cv2.dft(np.float32(img[:, :, x]), flags=cv2.DFT_COMPLEX_OUTPUT) for x in range(3)]

dft_shift = [np.fft.fftshift(dft[x]) for x in range(3)]

magnitude_spectrum = [20 * np.log(cv2.magnitude(dft_shift[x][:, :, 0], dft_shift[x][:, :, 1])) for x in range(3)]

# building RGB
shape = list(magnitude_spectrum[0].shape)
shape.append(1)
shape = tuple(shape)

RGBFt = np.concatenate((np.reshape(magnitude_spectrum[0], shape), np.reshape(magnitude_spectrum[1], shape),
                        np.reshape(magnitude_spectrum[2], shape)), axis=2)

plt.subplot(221), plt.imshow(img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(RGBFt.astype('uint8'), cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

rows, cols, channels = img.shape
crow, ccol = int(rows / 2), int(cols / 2)

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols, 2), np.uint8)
mask[crow - 60:crow + 60, ccol - 60:ccol + 60] = 1

# apply mask and inverse DFT
fshift = [dft_shift[x] * mask for x in range(3)]
f_ishift = [np.fft.ifftshift(fshift[x]) for x in range(3)]
img_back = [cv2.idft(f_ishift[x]) for x in range(3)]
img_back2 = [cv2.magnitude(img_back[x][:, :, 0], img_back[x][:, :, 1]) for x in range(3)]

# building RGB
filtered_image = np.concatenate(
    (np.reshape(img_back2[0], shape), np.reshape(img_back2[1], shape), np.reshape(img_back2[2], shape)), axis=2)
bw_image=np.sqrt(img_back2[0]**2+img_back2[1]**2+img_back2[2]**2)


plt.subplot(223), plt.imshow(mask[:, :, 0], cmap='gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(filtered_image.astype('uint8'), cmap='gray')

"""
Valamiért nem megy RGBben a dolog, csak külön-külön a channelek, ötlet erre?
"""

#plt.subplot(224), plt.imshow(bw_image, cmap='gray')
plt.title('Filtered image'), plt.xticks([]), plt.yticks([])
plt.show()
