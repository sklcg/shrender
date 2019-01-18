import sys
import numpy as np
import cv2 as cv

def load(filename):
    with open(filename) as file:
        height, width, n = map(int, file.readline().split())
        shmap = np.zeros((height, width, 16))
        i = 0
        for line in file.readlines():
            coeff = list(map(float, line.split()[1:]))
            if len(coeff) > 0:
                shmap[i // width][i % width] = np.array(coeff)
            i += 1
    return shmap

if len(sys.argv) != 3:
    exit('Usage: python3 convert.py xxx.shmap xxx.npy')

# convert
shmap = load(sys.argv[1])
np.save(sys.argv[2], shmap)
print('saving', sys.argv[2], 'done')

# visualize
shmap = np.sum(shmap, axis=-1)
image = (shmap - shmap.min()) / (shmap.max() - shmap.min())

# show
cv.imshow('image',image)
cv.waitKey(0)