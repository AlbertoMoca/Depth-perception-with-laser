import numpy as np
import matplotlib.pyplot as plt

from image_processing import ImageProcessing as im_p

import cv2
import time

import threading
import pickle
import queue

import json



cv2.namedWindow("original")
cv2.namedWindow("calibration")
cv2.namedWindow("borders")

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


def wait_order(q):
	while True:
		leng = input("Length:")
		q.put(leng)
		print("\n")

q = queue.Queue()
input_thread = threading.Thread(target= wait_order, args=(q,), daemon=True)
input_thread.start()

data_points = {}

while rval:

	
	mask = np.sum((np.array(frame)-np.array([150,150,150]))**2,axis=-1)**.5

	th = 70
	mask[mask<th]=0
	mask[mask>th]=1
	
	
	cv2.imshow("calibration", (mask*250).astype(np.uint8))

	borders = i_p.find_border(mask)
			
	cv2.imshow("borders", (borders * 200).astype(np.uint8))


	# Chain code
	figures, chains = i_p.chain_code_seg(borders,min_chain_size=10)
	centers_im = frame.copy()
	centers = []
	for figure,chain in zip(figures, chains):
		# Find center and sign

		center = i_p.centroid_figure(figure)
		centers.append(center)
		sign = i_p.figure_sign(center,chain)
		centers_im = cv2.circle(centers_im, (center[1],center[0]), radius=5, color=(0,0,0), thickness=3)

	centers = np.array(centers)
	corner_r = centers[np.argmax(centers[:,1])]
	corner_l = corner_r.copy()
	corner_l[1] = frame.shape[1]//2 - (corner_r[1]-frame.shape[1]//2 )
	
	centers_im = cv2.circle(centers_im, (corner_l[1],corner_l[0]), radius=5, color=(250,0,250), thickness=3)

	centers_im = cv2.circle(centers_im, (frame.shape[1]//2,frame.shape[0]//2), radius=5, color=(0,250,250), thickness=3)
	cv2.imshow("original", centers_im.astype(np.uint8))


	if not(q.empty()):
		dist = int(q.get())
		data_points[dist] = centers


	for i in range(2):
		rval, frame = vc.read()

	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
		break

with open('data.p', 'wb') as fp:
    pickle.dump(data_points, fp, protocol=pickle.HIGHEST_PROTOCOL)