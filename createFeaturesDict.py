import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import time
from os import walk
import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from imagenet_utils import preprocess_input
from os import walk
import random

"""
Compute features for segmented shots. Computes features for each frame individually and add all these features for one shot in one n-d array

Example usage:
python createFeaturesDict.py segmentedShotsFolder shotsLabelsFile
"""

def showFrames(allImages):
	for l in range(0,len(allImages)):
		_image = allImages[l]
		cv2.imshow('ImageWindow', _image)
		if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
			break

#read input
segmentedShotsFolder = str(sys.argv[1])
labelsFile = str(sys.argv[2])

#create pre-trained model
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)	#get the second last layer called FC1 of the model.

#add folders in a list
foldersPath = segmentedShotsFolder
folderNames = []
for(_,directories,_) in walk(foldersPath):
	folderNames.extend(directories)

folderNames = list(sorted(folderNames))
if os.path.isfile(labelsFile) == False:
	print("Labels file not found, exiting ...")
	sys.exit()

#read labels and add them in a dictionary. The key of dictionary is the folder number and the value is shot label
flabels = open(labelsFile,"r").readlines()
labelsDict = {}
for l in flabels:
	l = l.strip("\n").split(",")
	labelsDict[int(l[0])] = int(l[1])

features = []
labels = []
stoppingPoint = 10000
featuresDict = {}
folderCount = 0
for f in folderNames:
	print('Folder no',f)

	#if folder is not labeled and is not present in the labels file, skip it
	if int(f) not in labelsDict:
		continue

	#get label from dictionary
	label = labelsDict[int(f)]

	#if label is NAN, skip
	if label == 0:
		continue

	#get all images
	path = foldersPath + "/" + str(f) 
	filenames = []
	for(_,_,x) in walk(path):
		filenames.extend(x)

	folderCount = folderCount + 1
	images = []
	tempFeatures = []

	#sort filenames so that the video is in proper sequence
	filenames = list(reversed(list(reversed(sorted([int(f.strip(".jpg")) for f in filenames])))[:120]))
	filenames = [str(f).strip("\n")+".jpg" for f in filenames]
	allImages = []

	#adjust step size such that the total number of frames are always equal
	stepSize = 0
	if len(filenames) < 40:
		stepSize = 2
	elif len(filenames) < 60:
		stepSize = 3
	elif len(filenames) < 80:
		stepSize = 4
	else:
		stepSize = 6

	#total frames to keep
	totalLength = 20
	for m in range(0,len(filenames),stepSize):
		fil = filenames[m]
		path = foldersPath + "/" + str(f) + "/" + str(fil)

		#process the image for our convolutional neural network
		img = image.load_img(path, target_size=(224, 224))
		img = image.img_to_array(img)
		img = np.expand_dims(img, axis=0)
		img = preprocess_input(img)
		img = np.reshape(img,(224,224,3))

		#add all the images of one shot in this list
		allImages.append(img)
		
	if len(allImages) >= totalLength:
		allImages = allImages[len(allImages)-totalLength:]

	while len(allImages) < totalLength:
		allImages.append(np.zeros((224,224,3)))
	
	allImages = np.array(allImages)
	start = time.time()

	#get the features for all of the frames of our shot from the pre-trained convolutional neural network and save them in a list
	tempFeatures = model.predict(allImages)

	now = time.time()
	print('--Time taken and features shape',now-start,tempFeatures.shape)

	#save the features in a numpy dictionary
	featuresDict[str(f)+"-"+str(label)] = tempFeatures
	labels.append(label)

	#after every 100 features, save them in an incremental file
	if len(labels) % 100 == 0:
		np.save('shotsDict_'+str(len(labels))+'.npy', featuresDict) 

	#if enough data points have passed and we want to stop, stop
	if len(labels) > stoppingPoint:
		break

np.save('shotsDict_'+str(len(labels))+'.npy', featuresDict) 
