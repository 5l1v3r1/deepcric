# DeepCric
Deep Learning applied on Cricket Videos

![Screenshot](https://github.com/faizann24/deepcric/blob/master/Extra%20Data/architecture_shot.jpg)

## Description
The goal of this project is to apply deep learning models on cricket videos to generate automatic commentary for the videos. For this, we are using convolutional neural networks and long short term memory networks to first extract features from shots and then label them.

## Running the code
The code consists of several modules. Details for the modules are given below.

#### Scene Segmentation
This module segments scenes from an input video into different folders.

```Example Usage: python sceneSegmentation.py videoPath outputFolderPath```

After this module has completed, the 'outputFolderPath' will contain a large number of folders where each folder will be a scene.

#### Labeling Shots
This modules shows each scene segmented in the last step to the user and asks him/her to label this. This is a one time procedure and once we have around 1-2k scenes labeled, we use them to predict the new ones.

```Example Usage: python handLabelShots.py outputFile.csv folderWithSegmentedShots```

### Create Features Dict
This module takes each labeled scene labeled in the last step and extracts features of all the frames in a scene using a pre-trained Convolutional Neural Network (VGG19). The CNN is shown to extract global features from a frame. For each scene, we have the feature vector for each frame. We add all these feature vectors into one list. This list represents one complete scene. This is what we then pass to our final LSTM model.

```Example Usage: python createFeaturesDict.py folderWithSegmentedShots shotsLabelsFile```

This module will create dictionary files where the keys of dictionary are folder names and the values are the complete feature vectors for each scene/folder.

### Deep Learning Model
This is the module that we train ourselves. It has a small Long Short Term Memory network that utilizes the features extracted in the last step to predict the label for each scene. LSTMs have shown to have great performance on sequence related tasks. Since a video is just a sequence of frames, it is a problem where LSTMs work very good.

```Example Usage: python deepLearningModel.py dictionaryFile```

### Show Predictions
This module will save all the predictions made after training the above model in a separate folder as videos. This is for illustration purposes.

```Example Usage: python showPredictions.py folderContainingSegmentedShots folderToStoreVideos```

