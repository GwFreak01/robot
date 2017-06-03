import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Uncomment the next line for use in a Jupyter notebook
#%matplotlib inline
import numpy as np
import cv2

import perspective_transform
import color_thresh
# Read in the same sample image as before
image = mpimg.imread('sample.jpg')

source = np.float32([[17.426, 139.03], [300.676, 139.539],
                     [200.061, 97.0006], [118.041, 95.9817]])
grid_dst = 5
bottom_offset = 5
print(image.shape[1]//2, image.shape[0])
destination = np.float32([[image.shape[1]/2 - grid_dst, image.shape[0] - bottom_offset],
                          [image.shape[1]/2 + grid_dst , image.shape[0] - bottom_offset],
                          [image.shape[1]/2 + grid_dst, image.shape[0] - (grid_dst*2) - bottom_offset],
                          [image.shape[1]/2 - grid_dst, image.shape[0] - (grid_dst*2) - bottom_offset]])

# Assume you have already defined perspect_transform() and color_thresh()
warped = perspective_transform.perspect_transform(image, source, destination)
colorsel = color_thresh.color_thresh(warped, rgb_thresh=(140, 140, 140))

# Plot the result
plt.imshow(colorsel, cmap='gray')
plt.show()

ypos, xpos = colorsel.nonzero()
plt.plot(xpos, ypos, '.')
plt.xlim(0, 320)
plt.ylim(0, 160)
plt.show()

def rover_coords(binary_img):
    # Extract xpos and ypos pixel positions from binary_img and
    # Convert xpos and ypos to rover-centric coordinates
    x_pixel = 0
    y_pixel = 0
    return x_pixel, y_pixel