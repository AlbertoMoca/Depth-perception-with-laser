import numpy as np
import matplotlib.pyplot as plt

from image_processing import ImageProcessing as im_p

import cv2
import time

import threading
import pickle
import queue

from angle_corretion import DeepSystem
from distance_correction import CameraDeformations


cam = CameraDeformations()
ds = DeepSystem()

cv2.namedWindow("original")
cv2.namedWindow("calibration")
cv2.namedWindow("borders")
cv2.namedWindow("laser")

#Capture Initial Frames
vc = cv2.VideoCapture(0)
base = []

if vc.isOpened(): 
	for i in range(15):
		rval, frame = vc.read()
		base.append(frame.copy())
else:
	rval = False

# Calculate background
base = np.array(base)
bg = np.mean(base, axis=0)

i_p = im_p()



while rval:


	image_center =  (frame.shape[0]//2,frame.shape[1]//2)

	mask = np.zeros(frame.shape[:2])
	laser = np.zeros(frame.shape[:2])
	
	bw = np.mean(frame,axis=-1)

	laser[bw<160] = 0
	laser[bw>160] = 1

	mask[bw<100] = 1
	mask[bw>100] = 0

	laser_pos = np.unravel_index(np.argmax(bw, axis=None), laser.shape)
	#print(laser_pos)

	cv2.imshow("laser", (bw).astype(np.uint8))
	
	deep = ds.get_deep(((image_center[0]-laser_pos[0])**2+(image_center[1]-laser_pos[1])**2)**.5)

	cv2.imshow("calibration", (mask*250).astype(np.uint8))

	borders = i_p.find_border(mask)
			

	cv2.imshow("borders",(borders*250).astype(np.uint8))

	cv2.imshow("original", frame.astype(np.uint8))
	
	# Chain code
	figures, chains = i_p.chain_code_seg(borders,min_chain_size=100)
	centers_im = frame.copy()
	centers = []

	if len(figures) == 1:
		for figure,chain in zip(figures, chains):
			# Find center and sign

			#cv2.imshow("borders", (figure*np.arange(640) * 250/640).astype(np.uint8))

			max_x = np.argmax(np.max((figure*np.arange(640)),axis=0))
			min_x = np.argmax(np.max(figure/(figure*np.arange(640)+.01),axis=0))
			max_y = np.argmax(np.max((figure*np.array([np.arange(480)]).T),axis=1))
			min_y = np.argmax(np.max(figure/(figure*np.array([np.arange(480)]).T+ .01),axis=1))
			#print(f"max_x:{max_x}")
			#print(f"max_y:{max_y}")
			#print(f"min_x:{min_x}")
			#print(f"min_y:{min_y}")
			
			


			#print(figure.shape)
			center = i_p.centroid_figure(figure)
			centers.append(center)
			sign = i_p.figure_sign(center,chain)
			centers_im = cv2.circle(centers_im, (center[1],center[0]), radius=5, color=(0,0,0), thickness=3)

			centers_im = cv2.circle(centers_im, (max_x,max_y), radius=5, color=(0,250,150), thickness=3)
			centers_im = cv2.circle(centers_im, (max_x,min_y), radius=5, color=(0,250,150), thickness=3)
			centers_im = cv2.circle(centers_im, (min_x,min_y), radius=5, color=(0,250,150), thickness=3)
			centers_im = cv2.circle(centers_im, (min_x,max_y), radius=5, color=(0,250,150), thickness=3)
		

			corners = np.array([
				[max_y,max_x],
				[min_y,max_x],
				[min_y,min_x],
				[max_y,min_x],
			])

			distances = np.sum((corners-np.array(image_center))**2, axis=-1)**.5
			angles = np.arctan((corners-np.array(image_center))[:,0]/(corners-np.array(image_center))[:,1])
			print(distances)
			for i in range(len(distances)):
				print(("::::::::::::::::",deep,distances[i]))
				distances[i]= cam.get_distance_adjust(deep,distances[i],50)
			distances*=.65
			#print(distances)
			#print(angles*180/(np.pi))
			print(f"cos 0:{distances[0]*np.cos(angles[0])}\ncos 1:{distances[2]*np.cos(angles[2])}\nsin 0:{distances[0]*np.sin(angles[0])}\nsin 1:{distances[2]*np.sin(angles[2])}")
			print(f"ancho:{distances[0]*np.cos(angles[0])+distances[2]*np.cos(angles[2])}, alto:{distances[0]*np.sin(angles[0])-distances[2]*np.sin(angles[2])}")

			print("_______________________________")
		


	centers_im = cv2.circle(centers_im, (image_center[1],image_center[0]), radius=5, color=(0,250,250), thickness=3)
	centers_im = cv2.circle(centers_im, (laser_pos[1],laser_pos[0]), radius=5, color=(0,250,250), thickness=3)
	cv2.imshow("original", centers_im.astype(np.uint8))
	#print(image_center)


	for i in range(15):
		rval, frame = vc.read()

	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
		break
	

#with open('data1.p', 'wb') as fp:
#    pickle.dump(data_points, fp, protocol=pickle.HIGHEST_PROTOCOL)