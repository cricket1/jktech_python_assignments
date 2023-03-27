import cv2

image = cv2.imread('colour-images.png')

print(image.shape)

cv2.imshow('jktraining', image)
cv2.waitKey(0)


(b, g, r) = image[20, 100] # accesses pixel at x=100, y=20
(b, g, r) = image[75, 25] # accesses pixel at x=25, y=75
(b, g, r) = image[90, 85] # accesses pixel at x=85, y=90


print('program end')