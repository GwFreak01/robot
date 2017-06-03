import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# Uncomment the next line for use in a Jupyter notebook
# This enables the interactive matplotlib window
#%matplotlib notebook
image = mpimg.imread('example_grid1.jpg')
plt.imshow(image)
plt.show()

bl = [17.426, 139.03]
br = [300.676, 139.539]
tl = [118.041, 95.9817]
tr = [200.061, 97.0006]

import cv2
import numpy as np

def perspect_transform(img, src, dst):
    # Get transform matrix using cv2.getPerspectivTransform()
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp image using cv2.warpPerspective()
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    # Return the result
    return warped

# Define source and destination points
source = np.float32([[17.426, 139.03], [300.676, 139.539],
                     [200.061, 97.0006], [118.041, 95.9817]])
grid_dst = 5
bottom_offset = 5
print(image.shape[1]//2, image.shape[0])
destination = np.float32([[image.shape[1]/2 - grid_dst, image.shape[0] - bottom_offset],
                          [image.shape[1]/2 + grid_dst , image.shape[0] - bottom_offset],
                          [image.shape[1]/2 + grid_dst, image.shape[0] - (grid_dst*2) - bottom_offset],
                          [image.shape[1]/2 - grid_dst, image.shape[0] - (grid_dst*2) - bottom_offset]])

warped = perspect_transform(image, source, destination)
# Draw Source and destination points on images (in blue) before plotting
cv2.polylines(image, np.int32([source]), True, (0, 0, 255), 3)
cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
# Display the original image and binary
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=40)

ax2.imshow(warped, cmap='gray')
ax2.set_title('Result', fontsize=40)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

####### SANGEET CODE ###########
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
# import cv2
#
# image = mpimg.imread('example_grid1.jpg')
#
# def perspect_transform(img, src, dst):
#
#     # Get transform matrix using cv2.getPerspectivTransform()
#     M = cv2.getPerspectiveTransform(src, dst)
#     # Warp image using cv2.warpPerspective()
#     # keep same size as input image
#     warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
#     # Return the result
#     return warped
#
# # TODO:
# # Define a box in source (original) and
# # destination (desired) coordinates
# # Right now source and destination are just
# # set to equal the four corners
# # of the image so no transform is taking place
# # Try experimenting with different values!
#
# dst_size = 5
# bottom_offset = 6
# source = np.float32([[15, 140], [118 ,95],[201, 96], [302, 141]])
#
# destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
#                   [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
#                   [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset],
#                   [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
#                   ])
#
# warped = perspect_transform(image, source, destination)
# # Draw Source and destination points on images (in blue) before plotting
# cv2.polylines(image, np.int32([source]), True, (0, 0, 255), 3)
# cv2.polylines(warped, np.int32([destination]), True, (0, 0, 255), 3)
# # Display the original image and binary
# f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6), sharey=True)
# f.tight_layout()
# ax1.imshow(image)
# ax1.set_title('Original Image', fontsize=40)
#
# ax2.imshow(warped, cmap='gray')
# ax2.set_title('Result', fontsize=40)
# plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)