import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import time
from os import walk
import os
from scipy import spatial
import random

"""
Example Usage:
python sceneSegmentation.py videoPath outputFolderPath

Example: python sceneSegmentation.py video.mp4 SceneSegmentation (All separated scenes will be created inside 'SceneSegmentation')
"""

#taking video as input
videoPath = str(sys.argv[1])
cap = cv2.VideoCapture(videoPath)

#arrays to store distances
allFeatures = []
allDistances = []
allTimeSteps = []
folderNo = 1 #initial folder number in output folder
frameCount = 0

outputFolderPath = str(sys.argv[2])
os.system('mkdir "' + str(outputFolderPath) + "/"+str(folderNo)+'"')
while(True):

	#count frames, just for debugging purposes
	if frameCount % 1000 == 0:
		print('-- Frames traversed',frameCount)
	frameCount = frameCount + 1

	#read next frame
	ret, frame = cap.read()

	#calculate colored histogram of the frame
	color = ('b','g','r')
	hist = []
	for channel,col in enumerate(color):
		histr = cv2.calcHist([frame],[channel],None,[256],[0,256])
		hist.extend(histr)

	#resize frame - for fast operations
	frame = cv2.resize(frame,(0,0),fx=0.6,fy=0.6)
	cv2.imwrite(str(outputFolderPath)+"/"+str(folderNo)+"/"+str(frameCount)+".jpg",frame)
	
	#start checking for scenes after skipping the first three scenes since we need the first three frames as a reference
	if len(allFeatures) > 4:
		hist = np.array(hist)

		#average the last three frames
		prevThree = np.array(allFeatures[-1]) + np.array(allFeatures[-2]) + np.array(allFeatures[-3])
		
		#calculate distance of current frame with last three frames
		distance = spatial.distance.cosine(hist, prevThree)
		distance = distance * 100
		allDistances.append(distance)
		allTimeSteps.append(frameCount)

		#if distance of current frame is larger than the distance of mean of last three frames, we have a new scene
		if distance > 15.0:
			folderNo = folderNo + 1
			os.system('mkdir "' + outputFolderPath +"/" + str(folderNo) + '"')
	
	allFeatures.append(hist)