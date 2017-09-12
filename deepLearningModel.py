import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import time
from os import walk
import time
import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras import optimizers
from imagenet_utils import preprocess_input
import numpy as np
from scipy.spatial import distance
from os import walk
from keras.layers import Dense, Dropout, Activation,Flatten,Bidirectional,Input,Reshape,GRU,LSTM
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import random
from keras.utils import np_utils

"""
Get a features dictionary of shots and create a deep learning model to train on and classify the shots

Example usage:
python deepLearningModel.py dictionaryFile
"""

#inputs
dictionaryFile = str(sys.argv[1])

#load dictionary
featuresDict = np.load(dictionaryFile).item()

#make features and labels list to feed to the model
features = []
labels = []
labelFolders = []
for f in featuresDict:
	key = int(f.split("-")[1])
	if key == 1 or key == 2 or key == 3 or key == 4 or key == 5:
		features.append(featuresDict[f])
		labels.append(key)
		labelFolders.append(f)

features = np.array(features)
#convert labels to one hot encoding form. keras needs this
labels = np_utils.to_categorical(labels, None)
labels = np.array(labels)


#divide data into train and test split
randomState = random.randrange(0,100)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.10,random_state=randomState)
_, foldersTest, _, _ = train_test_split(labelFolders, labels, test_size=0.10,random_state=randomState)
inputShape = list(X_train.shape)[1:]
print('Input size is',features.shape,inputShape)

#optimizer configuration
adam = optimizers.Adam(lr=0.00001)

#MODEL ARCHITECTURE
model = Sequential()
model.add(Bidirectional(LSTM(256,implementation=0,kernel_initializer='random_uniform',return_sequences=False),input_shape=X_train.shape[1:]))
model.add(Dropout(0.3))
model.add(Dense(512,kernel_initializer='random_uniform'))
model.add(Dense(len(labels[0]),activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
print(model.summary())


#Running the model
epochs = 5
bestTestAccuracy = 0.0
bestTrainAccuracy = 0.0
resultsFile = open("resultsModel.csv","a")
for e in range(0,epochs):
	print('-- Iteration',e)
	score = list(model.evaluate(X_test,y_test))
	scoreTrain = list(model.evaluate(X_train,y_train))
	accuracyTest = score[1]
	accuracyTrain = scoreTrain[1]
	if accuracyTest > bestTestAccuracy:
		bestTestAccuracy = accuracyTest
	if accuracyTrain > bestTrainAccuracy:
		bestTrainAccuracy = accuracyTrain
	predictions = list(model.predict(X_test))
	if e == epochs-1:
		for p in range(0,len(predictions)):
			resultsFile.write(str(foldersTest[p])+","+str(''.join([str(e)+"," for e in predictions[p]]))+","+str(''.join([str(e)+"," for e in y_test[p]]))+"\n")

	print('Test and train accuracy is',accuracyTest,accuracyTrain)
	#resultsFile.write("Accuracy accuracy are"+str(accuracyTest)+","+str(accuracyTrain)+"\n")
	#resultsFile.write("MOVING NEXT\n")
	model.fit(X_train, y_train,batch_size=6,epochs=1,verbose=2,shuffle=True)

print('Highest accuracy obtained till now',bestTestAccuracy,bestTrainAccuracy)