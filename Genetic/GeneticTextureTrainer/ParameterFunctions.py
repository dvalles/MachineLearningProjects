
"""
This file contains all the functions to set genetic texture parameters
Also contains the image generator object
"""

import random
import math
import numpy as np
from DataStructures import Vector2, Vector3
import noise

#We'll pass this around through genetic functions, the image generator
class ImageGenerator:

	params = {}

	def __init__(self):
		self.params = {}
		for key, value in paramInstFunctions.items():
			self.params[key] = value()

def createImage(imageGenerator, size): #This is going to be where most of the work happens upon parameter changes
	image = np.zeros((size,size))
	for y in range(size):
		row = np.zeros((size))
		for x in range(size):
			uv = Vector2(x/size, y/size)
			uv *= 4 
			value = noise.pnoise2(uv.x, uv.y)
			value = max(0, value*255)
			row[x] = int(round(value))
		image[y] = (row)
	return image

def setFreq():
	return random.random()*20

def setOffset():
	return random.random()

#parameter instantiate functions dict
paramInstFunctions = {'frequency' : setFreq, 'offset' : setOffset}

#functions attached to different parameters (for later)
paramFunctions = {}

