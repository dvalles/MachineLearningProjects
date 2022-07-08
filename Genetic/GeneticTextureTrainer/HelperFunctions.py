"""
Some helper functions
"""

from PIL import Image
import math
import numpy as np
from os import listdir

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")

def resizeImageAndCrop(image, size = 100):
	minLength = min(image.size[1], image.size[0])
	image = image.crop((0,0,minLength, minLength))
	image = image.resize((size,size), Image.BICUBIC)
	return image

def resizeImage(image, size = 100):
	wpercent = (size/float(image.size[0]))
	hsize = int((float(image.size[1])*float(wpercent)))
	image = image.resize((size,hsize), Image.BICUBIC)
	return image

def prepareImages(imageSize = 100):
	folder = "TargetTextures/"
	pictures = []
	for file in listdir("TargetTextures/"):
		targetImage = Image.open(folder + file) #
		targetImage = resizeImageAndCrop(targetImage, imageSize)
		targetImage = targetImage.convert('L')
		targetImage = np.asarray(targetImage)
		pictures.append(targetImage)
	return np.asarray(pictures)

#clamp a number, because python doesn't have it
def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))