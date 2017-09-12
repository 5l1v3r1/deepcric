from os import walk
import cv2
import sys
import os

"""
Converts the predicted shots into videos and add text of predicted label on the video. This is just for illustration purposes.

Example Usage:
python showPredictions.py folderContainingSegmentedShots folderToStoreVideos

At the end, the videos are stored in the folderToStoreVideos.
"""

shotsDict = {0:"Shot",1:"Batsman",2:"Bowler",3:"Fielder",4:"Fielding"}
data = open('resultsModel.csv','r').readlines()
mainFolder = str(sys.argv[1])
resultsFolder = str(sys.argv[2])
if os.path.isfile(resultsFolder) == False:
	os.system("mkdir "+str(resultsFolder))
height = 432
width = 768
for d in data:
	d = d.split(",")
	folder = d[0].split("-")[0]
	predictions = [d[2],d[3],d[4],d[5],d[6]]
	actual = [d[8],d[9],d[10],d[11],d[12]]
	maxPred = predictions.index(max(predictions))
	maxPredValue = round(round(float(max(predictions)),3)*100,3)
	maxActual = actual.index(max(actual))
	predictedShot = shotsDict[maxPred]
	imageNames = []
	video = cv2.VideoWriter(str(resultsFolder)+str(folder)+"-"+str(predictedShot)+"-"+str(maxPredValue)+'.avi',cv2.VideoWriter_fourcc(*"XVID"),30,(width,height))
	for(_,_,filenames) in walk(mainFolder+"/"+folder):
		imageNames.extend(filenames)
	imageNames = list((sorted([s.split(".jpg")[0] for s in imageNames])))
	imageNames = [imageNames[i]+".jpg" for i in range(0,len(imageNames))]
	for i in range(0,len(imageNames)):
		image = cv2.imread(mainFolder+"/"+folder+"/"+imageNames[i])
		image = cv2.resize(image,(0,0),fx=2.0,fy=2.0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image,predictedShot,(0,50),font,1.5,(255,255,255),2,cv2.LINE_AA)
		video.write(image)
		cv2.imshow('Image',image)
		if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
			break
	video.release()
	
