import cv2

img = cv2.imread('data2/img1.png')
cv2.imshow('img1', img)
cv2.waitKey(0)
cv2.destroyAllWindows()