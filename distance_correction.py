import pickle
import numpy as np 
from scipy import interpolate

class CameraDeformations:
	def __init__(self):
		with open('data.p', 'rb') as fp:
			data = pickle.load(fp)

		deep = np.array(list(data.keys()))

		pixels = []
		for i in data:
			sample = data[i]
			sample = np.delete(sample, np.argmin(sample[:,1]), 0)
			origin = sample[np.argmax(sample[:,0])]
			sample = np.delete(sample, np.argmax(sample[:,0]), 0)
			pixels.append(np.sum((sample-origin)**2,axis=-1)**.5)

		x,y = (32,32)
		dist = []
		for i in range(1,10):
			dist.append(((i*x/6)**2+(i*y/6)**2)**.5)

		dist = np.array(dist[::-1])
		pixels= np.array(pixels)

		X = dist
		Y = deep
		X, Y = np.meshgrid(X, Y)

		Z = pixels

		self.f = interpolate.interp2d(X, Y, Z, kind = 'cubic')
		
	def get_distance_adjust(self,profundidad, pixels, detail = 30):
		
		return np.linspace(5,35,detail)[np.argmin(np.abs(self.f(np.linspace(5,35,detail), profundidad) - pixels))]