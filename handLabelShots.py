import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import time
from os import walk
import os

"""
This function takes the segmented shots and asks a user to label them

Example usage: python handLabelShots.py outputFile.csv folderWithSegmentedShots
"""

#take inputs
outputFile = str(sys.argv[1])
segmentedShotsFolder = str(sys.argv[2])

#read data if there is any
labelsDict = {}
if os.path.isfile(outputFile) == True:
	flabels = open(outputFile,"r").readlines()
	for l in flabels:
		l = l.strip("\n").split(",")
		labelsDict[int(l[0])] = l[1]

#read folder names
foldersPath = segmentedShotsFolder
folderNames = []
for(_,directories,_) in walk(foldersPath):
	folderNames.extend(directories)

#sort folder names
folderNames = list((sorted([int(f) for f in folderNames])))
labelFile = open(outputFile,"a")

#dictionary to map numbers to shot names
shotDict = {0:"nan",1:"shot",2:"batsman",3:"bowler",4:"fielder",5:"fielding",6:"crowd",7:"scorecard",8:"empire"
			,9:"bowlerrunning",10:"aftershot",11:"diffplayers",12:"Replay"}

for x in range(0,len(folderNames)):
	f = folderNames[x]
	print('--Folder',f)
	try:
		#if we have this folder labeled, continue to next folder
		if int(f) in labelsDict:
			continue
		path = foldersPath + "/" + str(f) 
		filenames = []
		for(_,_,x) in walk(path):
			filenames.extend(x)

		#if folder contains very few frames, skip it.
		if len(filenames) < 10:
			continue

		#get all images from folder and show them to the user in the form of a video
		images = []
		filenames = list(sorted([int(f.split(".jpg")[0]) for f in filenames]))
		filenames = [str(f)+".jpg" for f in filenames]
		fcount = 0
		for z in range(0,len(filenames)-2):
			fil = filenames[z]
			fcount = fcount + 1
			if fcount % 3 != 0:
				continue
			path = foldersPath + "/" + str(f) + "/" + str(fil)
			im = cv2.imread(path)
			cv2.imshow('video',im)
			if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
				break

		#ask the user for the label
		name = input("""0 - Nan\n1 - Shot\n2 - Batsman\n3 - Bowler\n4 - Fielder\n5 - Fielding\n6 - Crowd\n7 - Scorecard\n8 - Empire\n9 - Bowler Running\n10 - After Shot\n11 - Diff Players\n12 - Replay\nEnter label: """)
		name = int(name)

		#print and store the label
		print(f,shotDict[name])
		labelFile.write(str(f)+","+str(name)+"\n")
	except Exception as e:
		print(e)

'''
0 - Nan (Cannot decide the label)
1 - Shot
2 - Batsman
3 - Bowler
4 - Fielder
5 - Fielding
6 - Crowd
7 - Scorecard
8 - Empire
9 - Bowler Running
10 - After Shot
11 - Diff Players
12 - Replay
'''