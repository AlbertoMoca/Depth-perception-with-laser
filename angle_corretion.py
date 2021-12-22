import pickle
import numpy as np 
from scipy import interpolate
from scipy.interpolate import interp1d

class DeepSystem:
	def __init__(self):
		with open('distance_angle_map.p', 'rb') as fp:
			data = pickle.load(fp)
		
		dist = np.array(list(data.keys())).astype(float)
		pixels = np.array(list(data.values())).astype(float)
		
		th = np.arctan(pixels/dist)
		
		self.f = interp1d(pixels, th, kind='cubic',bounds_error=False)
	
	def get_deep(self,pixels):
		return pixels/np.tan(self.f(pixels))