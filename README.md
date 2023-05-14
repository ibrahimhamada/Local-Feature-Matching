# Local-Feature-Matching
Local Feature Matching:  Mini Project 2 of the Computer Vision Course Offered in Spring 2022 @ Zewail City

The project is adopted from a similar course in Brown University.
The goal is to create a local feature matching algorithm and attempt to match multiple views of real-world scenes using  Harris Corner Detector and SIFT.

## Dataset:
The dataset consists of six images that can be viewed as three pairs of images, which include:

* Notre Dame de Paris
* Mount Rushmore
* Gaudi's Episcopal Palace

The following functions are used in our implementation Local Feature Matching:

   1) `get_interest_points()`: Harris Corner Detector to get the points of interest after obtaining the gradient of the image and removing the noise from the image using Gaussian Filter.
    
   2) `get_features()`: It is a SIFT-like Local Feature Descriptor to get the features from the image.
    
   3) `match_features()`: It is implemented using the nearest neighbor distance ratio test method of matching local features. The Euclidean Distance is found and then the nearest neighbor is obtained
 

## Results

### Mount Rushmore

![CIE552_SPRG2022_Mini-project_2_is it in the pic_mt_rushmore_matches](https://github.com/ibrahimhamada/Local-Feature-Matching/assets/58476343/fe8fc778-1f36-4f9b-a57f-dfdc69809002)

Accuracy = 98%

### Notre Dame de Paris

![CIE552_SPRG2022_Mini-project_2_is it in the pic_notre_dame_matches](https://github.com/ibrahimhamada/Local-Feature-Matching/assets/58476343/0e191cc5-9c33-4398-a240-818721fde07b)

Accuracy = 98%
