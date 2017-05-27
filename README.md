Cross Device Tracking

Settings:
=========

This software contains two setting files:

'Variables.py' contains the environment variables that the software needs:

 * There is an environment variable for the absolute path of every file provided.
 * There is an environment variable for the python wrapper folder of XGBoost.
 * There is an environment variable for the path to save/load the model.

'VariablesTST.py' contains two environment variables:

 * There is an environment variable for the absolute path of the test file.
 * There is an environment variable for the absolute path of the file with the results.

Running the code:
=================

To train the algorithm and create the model:

    python train.py

To use the model and make the predictions on a test file:

    python predict.py


Description of the algorithm:
=============================

* Preprocessing
At the initial stage, we iterate over the list of cookies looking for other cookies with the same handle. Then, for every pair of cookies with the same handle, if one of them doesn't appear in an IP address that the other cookie appears, we include all the information about this IP address in the cookie.

* Initial selection of candidate cookies for every Device:
It is not possible to create a training set containing every combination of devices and cookies due to the high number of them. In order to reduce the initial complexity of the problem and to create an affordable dataset, some basic rules have been created to obtain an initial reduced set of candidate cookies for every device. The rules are based on the IP addresses that both device and cookie have in common and how frequent they are in other devices and cookies. The procedure, for every device, is as follows.

  1. We create a set that contains the device's IP addresses that appear in less than ten devices and less than twenty cookies. The initial list of candidates is every cookie with known handle that appears in any of theses IP addresses.
  1. If the previous rule returned an empty set of candidates we create a set that contains the device's IP addresses that appear in less than twenty five devices and less than fifty cookies. The initial list of candidates is every with known handle cookie that appears in any of theses IP addresses.
  1. If the previous rule returned an empty set of candidates we create a set that contains the device's IP addresses. The initial list of candidates is every cookie with known handle that appears in any of theses IP addresses.
  1. If the previous rule returned an empty set of candidates we create a set that contains the device's IP addresses. The initial list of candidates is every cookie that appears in any of theses IP addresses.
  1. If a cookie has the same handle than any of the candidates then this cookie is a candidate too.

* Creating the datasets:
Every pattern in the training and test set represents a device/candidate cookie pair obtained by the previous step and contains information about the device (Operating System (OS), Country, ...), the cookie (Cookie Browser Version, Cookie Computer OS,...) and the relation between them (number of IP addresses shared by both device and cookie, number of other cookies with the same handle than the cookie,...).

* Training procedure (Supervised Learning + Bagging)
To create the classifier, we have selected a Regularized Boosted Trees algorithm. The software that we used was XGBoost.
We have used 8 baggers creating 8 different subdatasets from the original dataset.

* Semi Supervised Learning:
Semi-supervised learning is a class of supervised learning that also make use of unlabeled data. In our case we make use of the data contained in the test set. If we sort the scores obtained by every candidate and the first score is high and the second is very low, is very likely that the first cookie belongs to the device. We make use of this information to update some features in the training set and retrain the algorithm again.

* PostProcessing:
We iterate over the devices using the following procedure:
If the initial selection of candidates did not find a candidate with enough likelihood (logistic output of the classifier) we choose a new set of candidate cookies selecting every cookie that shares an IP address with the device and we score them using the classifier.
We label the cookie with highest score as one of the device's cookies. If there are other cookies with the same handle than this cookie we label them too.
We sort the candidates in descending order by the score they have reached and we iterate over them. We label them as a device's cookie if they reach a threshold.
The value of the threshold changes attending to:
 The number of cookies already labeled as device's cookies.
 The number of other cookies with the same handle than this one.
 The handle of the cookie is known or not.
 The of the best candidate is known or not.
