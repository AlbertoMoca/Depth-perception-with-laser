import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from scipy import signal


import cv2

from PIL import Image


class ImageProcessing():
	def __init__(self):
		self.kmeans = None

	
	def initial_kmean_segmentation(self,image, n_clusters):
		im_raw = np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2]))
		self.kmeans = KMeans(n_clusters=n_clusters).fit(im_raw)
		im_seg = np.reshape(self.kmeans.labels_,(image.shape[0],image.shape[1]))


	def segmentation(self, image):
	
		seg = self.kmeans.predict(np.reshape(image,(image.shape[0]*image.shape[1],image.shape[2])))
		seg = np.reshape(seg,(image.shape[0],image.shape[1]))

		return(seg)
	
	def find_border(self, seg):
		f = np.array([[1,1,1],[1,1,1],[1,1,1]])
		imp = cv2.filter2D(seg.copy().astype(np.uint8),-1,f).astype(np.float)
		imp[(imp < 5) & (imp >= 2)] = 250
		imp[imp != 250] = 0
		imp[imp == 250] = 1

		return(imp)
	

	def chain_code_seg(self, borders, min_chain_size = 30):
		visual_mask = np.zeros(borders.shape)
		figures = []
		chains = []
		circular_order = np.array([
			[0,1],[0,2],[1,2],[2,2],[2,1],[2,0],[1,0],[0,0]
		])
		figure_id = 0 
		while True:
			
			figure = np.zeros(borders.shape)
			chain = []

			xs,ys = np.where(borders == 1)
			if len(xs) == 0:
				break
			x0,y0 = xs[0],ys[0]
			
			while True:
				
				
				borders[x0,y0] = 0
				figure[x0,y0] = 1
				chain.append([x0,y0])

				mask = borders[x0-1:x0+2,y0-1:y0+2]
				end_flag = True
				
				

				for j,i in enumerate(circular_order):
					
					if mask.shape != (3,3):
						end_flag = True
						break
					if mask[i[0],i[1]] == 1:
						x0 += i[0] - 1
						y0 += i[1] - 1
						end_flag = False
						circular_order = np.roll(circular_order, -j+4,axis=0)
						break
				
				#plt.imshow(figure)
				#plt.show()
				if end_flag:
					if min_chain_size < len(chain):
						chain.append([x0,y0])
						chains.append(chain)
						figures.append(figure)
					break
			figure_id+=1
			
		return (figures,chains)


	def centroid_figure(self, mask):
		xs,ys = np.where(mask == 1)
		center = (int(xs.mean()),int(ys.mean()))
		return center
	
	def figure_sign(self, center, chain, plot=False):
		a = (np.array(chain)-np.array([center[0],center[1]]))**2
		sign = (a[:,0] + a[:,1])**.5
		sign /= max(sign)
		
		x = np.linspace(0, len(sign), len(sign))
		xvals = np.linspace(0, len(sign), 50)
		sign = np.interp(xvals, x, sign)
		
		if plot:
			plt.plot(sign)
			plt.ylim(0, 1.1)
			plt.show()
		return sign
		