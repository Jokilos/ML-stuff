import cv2
import numpy as np

def find_color(imgpath):
    image = cv2.imread(imgpath)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h_eps = 10
    sv_eps = 50
    eps = np.array([h_eps, sv_eps, sv_eps])
    def pick_color(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  
            hsv_pixel = hsv_image[y, x]  
            
            lower_bound = np.clip(hsv_pixel - eps, 0, 255)
            upper_bound = np.clip(hsv_pixel + eps, 0, 255)
            
            print(f"lower1 = np.array({lower_bound})")
            print(f"upper1 = np.array({upper_bound})")

    cv2.imshow('Image', image)
    cv2.setMouseCallback('Image', pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

find_color('task2_imgs/wall_hitting.png')