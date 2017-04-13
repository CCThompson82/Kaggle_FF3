# Nature Conservancy Fisheries Monitoring Kaggle Competition
Christopher C Thompson 

## Competition and Data
The details of this competition can be found on the [Kaggle Competition](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring) page.  The data is available to [download](https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data).

## Objective
The purpose of this competition is to develop a model that can identify various species of fish from images captured from elevated cameras
on board fishing vessels.  

## Data Set Exploration
A detailed analysis of the data set and label classes can be found in the 
['Fishery Data Exploration and Preprocessing.ipynb'](https://github.com/CCThompson82/Kaggle_FF3/blob/master/Fishery%20Data%20Exploration%20and%20Preprocessing.ipynb)
file in this repository.  Important aspects of the data include :

  * 8 mutually exclusive fish classes 
  * 3777 training set images - stored in separate directories for each class
  * 1000 stage 1 test set images (no label provided)
  * 12153 stage 2 test set images (no label provided) - available only during the last 7 days of the competition.

All training set images were medium resolution, with varying dimensions and aspect ratios.  In general, images capture the entire deck that 
may or may not contain a fish, fisherman, equipment, etc, and were taken during both night and day.  

## Strategy
While seemingly a straight forward computer-vision object classification task, there were several complicating factors that made this 
competition interesting.  The major issue was the complicated composition of each image (with people moving, 
different boats, equipment, partial views of fish, etc).  As such, just 3777 images, of which the set was severely imbalanced with some fish
classes representing less than 5% of the total, was to be considered an undersized dataset.  Moreover, it was clear that in order for the model to generalize
in the future, the model would need to accommodate images from unseen boats and camera angles and heights.  

With this in mind, my main concern was to keep the model from exhibiting high variance, even at the expense of predictive bias in the training 
phases.

### [FishFinder](https://github.com/CCThompson82/Kaggle_FF3/tree/master/FishFinder) 
My model was composed of two neural networks.  The first, named 'FishFinder' received a coarse image (downsized to a standard pixel dimension 
of 64x112x3), and output the probability that a fish was contained in the image, along with specifications for making a bounding box for
a region of interest.  The box predictions were a scale value, along with proportional coordinates for the top left corner of a bounding box.
Using these specifics, a high resolution image could be scaled, an anchor dropped at the yx coordinate, and a standard pixel length
used to generate a uniform sized fovea crop containing the region of interest.  Labels for the Fish/No Fish ('FiNoF') were known (NoF : 0, else : 1),
however bounding boxes required manual annotation.  Instead of manually annotating every one of the training images in order to provide
a training set for bounding box prediction, I utilized multi-task learning.

In brief, the idea was to primarily train a network to predict the known label of whether a fish was present or absent from a coarse image.
However, the last layer prior to FiNoF prediction was bifurcated to a separate classifying layer used to predict the box specifics.  Where
that coarse image had been annotated manually, cross entropy was measured against the label.  If the example had not been annotated, its 
contribution to the cross entropy score against a fake label was negated by 0 weight.  The cross entropy from the FiNoF classification is
combined with the cross entropy score from the box specifics classification, and used for gradient calculation and backpropigation.  Thus
the model learns primarily if a fish is present in the image, but also occassionally how to predict where the fish may be.  These tasks
are related.  After sufficient training, the outputs from the model were used to generate semi-supervised targets for the box specifications
of images that had not been manually annotated.  By combining the tasks, the model learned where to place region of interest boxes without
explicit labels for the entire data set.  

### [FishyFish](https://github.com/CCThompson82/Kaggle_FF3/tree/master/FishyFish)
However, the FishFinder model was not sufficient to detect types of fish.  The coarse image generation was much too crude for 
detection of fish types by eye.  It is possible a good neural network could have succeeded at this task, but sometimes the fish 
would blend entirely into the background deck color.  To aid in classification of types of fish, the bounding box predictions 
generated from the FishFinder model, were used to generate region of interest fovea for every training and test set example.

These fovea were supplied to the neural network `FishyFish` that received as input: 

  * The final layer before bifurcation of FishFinder classifications (`coarse embedding`)
  * The probability of FiNoF 
  * 64x64x3 fovea of region of interest

The fovea was run through a convolutional network (see below), prior to transformation through fully connected layers.  In the penultimate
layer, the tensor is concatanted with the FiNoF probability and coarse embedding provided for the image.  This concatenation was then 
subjected to a final connected layer and then softmax classification versus the 8 fish classes in one hot label format.  

#### Notes on convolution
As I did not have time (or GPU power) to train a convolutional network from scratch, I initialized the convolutional layers with the 
first six layers of the [VGG-19 network](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) for both 
the FishFinder and FishyFish networks.  Early attempts allowed these layers to 
be trained via backpropigation, but I did not see any difference in learning by making these layers static, whereas the speed of training
increased dramatically.  Given more time (deadline was looming), I could have generated a embedding vector for every coarse image and stored 
the array in memory, which would have been significantly faster than loading the coarse image when called.  

## Results
The cross entropy for all targets (the two outputs for FishFinder and the actual classification in FishyFish) clearly decreased for both 
the train and validation (unseen) sets.  In fact the model was still learning when the deadline forced me to submit stage 2 predictions.

The final LogLoss score for the test set was 1.784, good enough for 121 position out of 2,293 teams entered.  Most entrants seemed to overfit
their model to the training data, from which the testing images were supremely different.  Thus it is probably the early cutoff of the FishyFish
model training was beneficial.  

## Discussion
While I achieved a great position in the competition, I do not think this model could be rolled out for the Nature Conservancy Fishery service.
If I could do this project again, given the same computing options (i.e. not hiring a high powered GPU instance), I would have used several 
of the well known ImageNet competition frameworks (Inception, VGG, etc) to generate embedding vectors that represent random cropped fovea
for each image, without training or updating the convolutional layers.  These embeddings would then be subjected to an autoencoder, from which
further compression could be acheived.  Finally, manually annotated fovea could be used for one-shot learning of what each fish looks like
in the generated manifold, and these characteristics could be used to generate a sufficient number of semi-supervised training examples that 
could be used to train a standard deep network for pixel by pixel classification.  This would be expensive computationally, but is the only
way that I can think of to confidentally predict fish class from an undersized, imbalanced, **non-representative** set of training images that
would generalize to completely new boats and cameras.  


