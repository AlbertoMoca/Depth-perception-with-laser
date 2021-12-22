import numpy as np
import matplotlib.pyplot as plt

import cv2
import time

import threading
import pickle
import queue



cv2.namedWindow("original")
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

	laser = np.zeros(frame.shape[:2])
	
	bw = np.mean(frame,axis=-1)
	th = 170
	laser[bw<th] = 0
	laser[bw>th] = 1
	cv2.imshow("laser", (laser*250).astype(np.uint8))

	laser_pos = np.unravel_index(np.argmax(laser, axis=None), laser.shape)
	#print(laser_pos)
	
	image_center =  (frame.shape[0]//2,frame.shape[1]//2)

	centers_im = frame.copy()
	centers_im = cv2.circle(centers_im, (image_center[1],image_center[0]), radius=5, color=(0,250,250), thickness=3)
	cv2.imshow("original", centers_im.astype(np.uint8))

	h = ((image_center[0]-laser_pos[0])**2+(image_center[1]-laser_pos[1])**2)**.5

	if not(q.empty()):
		dist = int(q.get())
		data_points[dist] = h


	for i in range(15):
		rval, frame = vc.read()

	key = cv2.waitKey(1)
	if key == 27: # exit on ESC
		break
	

with open('distance_angle_map.p', 'wb') as fp:
    pickle.dump(data_points, fp, protocol=pickle.HIGHEST_PROTOCOL)