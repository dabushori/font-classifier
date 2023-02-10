import numpy as np
import cv2 as cv

mean = np.loadtxt('data/char_mean.txt')
std = np.loadtxt('data/char_std.txt')

def img_transform(x):
    x = x.astype(np.float32).mean(axis=2) / 255
    
    filter = cv.getGaussianKernel(5, 0.5)
    x = cv.filter2D(x, -1, filter)
    return x
 
def char_transform(x):
    x = x.astype(np.float32)
    filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    x = cv.filter2D(x, -1, filter)    
    x = (x - mean) / std
    return np.array([x])

def labels_transform(font):
    label_idx = {b'Titillium Web': 0, b'Alex Brush': 1, b'Ubuntu Mono': 2, b'Open Sans': 3, b'Sansation': 4}[font]
    one_hot_encoded = np.zeros(5, dtype=np.float32)
    one_hot_encoded[label_idx] = 1
    return one_hot_encoded