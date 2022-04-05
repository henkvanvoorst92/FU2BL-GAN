
import numpy as np
from matplotlib import cm
import cv2


def add_text_to_img(img,text, scale=3, color=(1), org=(10,40), linetype=2, font=cv2.FONT_HERSHEY_PLAIN, thickness=2):
    #https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
    # img  --  a 2D array with image pixels
    # text --  string to stamp in the upper left corner
    img = cv2.putText(img = np.copy(img),
        text = text, 
        org = org, 
        fontFace=font, 
        fontScale = scale,
        color = color,
        lineType = linetype, 
        thickness=thickness)
    return img