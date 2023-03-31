import cv2

def isgray(imgpath):
    img = cv2.imread(imgpath)
    if len(img.shape) < 3: return True
    if img.shape[2]  == 1: return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b==g).all() and (b==r).all(): return True
    return False

print("*************************")
img_path = "color.jpeg"
image = cv2.imread(img_path)

print(image.shape)

cv2.imshow(img_path, image)
cv2.waitKey(5000)
print("Is GrayScale Image: "+ str(isgray(img_path)))
print("*************************")