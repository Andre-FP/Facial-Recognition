# Facial-Recognition
Polytech work in computer vision for facial recognition of all classmates 
of the course of computer vision. 

The DeepFace library was used to extract the embeddings, and classical classifier models
were evaluated to determine to which person the photo belongs.

![Example1](images-demo/me_identification.JPG)


## Dataset

A dataset with 10 photos of each person in the class, composed of 11 people, was used, 
with the face clearly visible in each photo.

## Implementation

For the implementation of the solution, a sequence of steps was necessary. At first, 
it was necessary to crop all the photos to only show the face of each person. Since the dataset 
is the ground truth, it was done by hand, with each person cropping their own photos so that this 
task wouldn't take too long.

The sequence of implementation steps consisted in extracting the embeddings of the image,
and training a classifier to determine who is the author of the photo.

### Embeddings Extraction

To extract the embeddings, the library DeepFace was used, that offers a framework
to facilitate the testing of differents models for embeddings extraction. This way, it was
tested the "VGG-Face" and "Facenet512" models.

### Face Classification

For the classification part, many models were tested to find the best one, and 
to compare the results: MLPClassifier, KNN, SVMClassifier, LinearDiscriminantAnalysis, 
QuadraticDiscriminantAnalysis and GaussianNB.

The model's input was the embeddings and the ground truth the person's name.

### Data Augmentation

To achieve good results, it was necessary to apply data augmentation techniques.
The reason is that the amount of data for each person is limited, and it doesn't cover
many possible scenarios that may occur in real time.

So, the technique applied was the variation of luminosity of the photo, which
improved the results considerably.

![Data Augmentation](images-demo/data_augmentation.JPG)



### Results

The results of each classifier, with VGG-Face and Facenet, are shown in the 
figure below. They were evaluated on the database that was split in 
train and test. 

![Example1](images-demo/results.JPG)


### Real time

In real time, the results are a little worse due to scenarios that are not covered by the dataset. 
However, the model still achieves good and consistent results.

To evaluate in real time, it was necessary to use an algorithm for face detection, to 
automatically crop the face for the embedding extractor and the face classificator.
To do this, the detection models already aggregated in DeepFace were used: "MTCNN" and the DeepFace implementation with OpenCV called "OpenCV"

The MTCNN achieved better results in detecting faces, but it slowed down the execution. In contrast, 
the solution with OpenCV was faster, but didn't perform very well. So, the solution with MTCNN was 
preferred because of its performance.
