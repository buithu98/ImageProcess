# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 15:30:31 2019

@author: Bui Thi Hoai Thu - Taos
@ Ma sv: 16021424
"""
import numpy as np
import cv2


def find(my_array, target):
    tmp = my_array - target
    mask = np.ma.less_equal(tmp, -1)
    
    if np.all(mask):
        c = np.abs(tmp).argmin()
        return c 
    masked_tmp = np.ma.masked_array(tmp, mask)
    return masked_tmp.argmin()
def match(img_start, img_match):
    length = img_start.shape
    #print(img_start.shape)
    img_start = img_start.ravel()
    img_match = img_match.ravel()

    s_values, bin_idx, s_counts = np.unique(img_start, return_inverse=True,return_counts=True)
    t_values, t_counts = np.unique(img_match, return_counts=True)
    #print(np.unique(img_start, return_inverse=True,return_counts=True))
    # Calculate s_k 
    s = np.cumsum(s_counts).astype(np.float64)
    s /= s[-1]
    t = np.cumsum(t_counts).astype(np.float64)
    t /= t[-1]
 
    # Round 
    sour = np.around(s*255)
    temp = np.around(t*255)
    b=[]
    for data in sour[:]:
        b.append(find(temp,data))
    b= np.array(b,dtype='uint8')
    return b[bin_idx].reshape(length)    
# load img in grayscale
img_start = cv2.imread('./img1.jpg',cv2.IMREAD_GRAYSCALE)
img_match = cv2.imread('./lena.jpg',cv2.IMREAD_GRAYSCALE)

img_transform = match(img_start, img_match)
#show img
cv2.imwrite('./img_transform.jpg', img_transform)
cv2.imshow('img_transform',img_transform)

cv2.waitKey(0)
cv2.destroyAllWindow()
