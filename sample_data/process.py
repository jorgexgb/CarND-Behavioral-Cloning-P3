import cv2
import numpy as np

center_img = cv2.imread('center_orig.jpg')
left_img = cv2.imread('left_orig.jpg')
right_img = cv2.imread('right_orig.jpg')

center_img = center_img[70:135, 0:320]
left_img = left_img[70:135, 0:320]
right_img = right_img[70:135, 0:320]

flipped_center_img = cv2.flip(center_img, 1)
flipped_left_img = cv2.flip(left_img, 1)
flipped_right_img = cv2.flip(right_img, 1)

cv2.imwrite('center_crop.jpg', center_img)
cv2.imwrite('center_flipped.jpg', flipped_center_img)
cv2.imwrite('left_crop.jpg', left_img)
cv2.imwrite('left_flipped.jpg', flipped_left_img)
cv2.imwrite('right_crop.jpg', right_img)
cv2.imwrite('right_flipped.jpg', flipped_right_img)

# crop image
