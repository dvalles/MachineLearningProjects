"""
This class provides helpful data structures not implemented in python
"""

class Vector3:
	x = 0
	y = 0
	z = 0

	def __init__(self, x = 0, y = 0, z = 0):
		self.x = x
		self.y = y
		self.z = z

	def __mul__(self, other):
		self.x *= other
		self.y *= other
		self.z *= other
		return self

class Vector2:
	x = 0
	y = 0

	def __init__(self, x = 0, y = 0):
		self.x = x
		self.y = y

	def __mul__(self, other):
		self.x *= other
		self.y *= other
		return self
