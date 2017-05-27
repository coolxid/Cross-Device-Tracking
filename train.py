print "Starting the Training"
import csv
import os
import re
import numpy as np
import inspect
import sys
import sklearn
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
import pickle
import pandas as pd
from Variables import *

import xgboost as xgb
from CookieLibrary import *

#################################################################################
# PARSING THE FILES PROVIDED FOR THE CHALLENGE AND CREATING THE DATA STRUCTURES #
# THAT THE ALGORITHM NEEDS                                                      #
#################################################################################

# Some features in the files that describe the cookies and the devices are categorical features in test mode.
# For example, the countries are like: 'country_147', or the handle is like 'handle_1301101'.
# This function creates dictionaries to transform that text into a numerical value to load them in a numpy matrix.

print('Loading Dictionaries')
(DeviceList, CookieList, HandleList, DevTypeList, DevOsList,ComputerOsList,ComputerVList,CountryList,annC1List,annC2List)=GetIdentifiers(trainfile,testfile,cookiefile)

DictHandle = list2Dict(HandleList)
DictDevice = list2Dict(DeviceList)
DictCookie = list2Dict(CookieList)
DictDevType = list2Dict(DevTypeList)
DictDevOs = list2Dict(DevOsList)
DictComputerOs = list2Dict(ComputerOsList)
DictComputerV = list2Dict(ComputerVList)
DictCountry = list2Dict(CountryList)
DictAnnC1 = list2Dict(annC1List)
DictAnnC2 = list2Dict(annC2List)

######################################################
#Splitting the Train File into 70% Train 30% Test Set#
######################################################

data=pd.read_csv(trainfilesplit)
dummyList=[1]*len(data)
DevicesTrain,DevicesTest,_,_=train_test_split( data, dummyList,test_size=0.3)

######################
#Writing to CSV FIle #
######################

DevicesTrain.to_csv('Data/DevicesTrain.csv',index = False) 
DevicesTest.to_csv('Data/DevicesTest.csv',index = False) 


# This part loads the content of the devices into a numpy matrix using the dictionaries to transform the text values into numerical values
print('Loading Devices Files')
X_Train = loadDevices(trainfile,DictHandle,DictDevice,DictDevType,DictDevOs,DictCountry,DictAnnC1,DictAnnC2)
#DevicesTest = loadDevices(testfile,DictHandle,DictDevice,DictDevType,DictDevOs,DictCountry,DictAnnC1,DictAnnC2)


#########################################################
#Splitting the X_train into 70% Train 30% Validation Set#
#########################################################
dummyList=[1]*len(X_train)
DevicesTrain,DevicesTest,_,_=train_test_split( X_train, dummyList,test_size=0.3)

# This part loads the content of the cookies into a numpy matrix using the dictionaries to transform the text values into numerical values
print('Loading Cookies File')
Cookies = loadCookies(cookiefile,DictHandle,DictCookie,DictComputerOs,DictComputerV,DictCountry,DictAnnC1,DictAnnC2)

# It loads the Properties of the devices
print('Loading Properties File')
DevProperties=loadPROPS(propfile,DictDevice,DictCookie)

# It read the train information and creates a dictionary with the cookies of every device, a dicionary that gives for every cookie the other cookies in its same handle and for every cookie its devices
(Labels,Groups,WhosDevice)=creatingLabels(DevicesTrain,Cookies,DictHandle)


# It creates a dictionary whose keys are the ip address and the value a numpy array with the IP info
print('Loading IP Files')
XIPS=loadIPAGG(ipaggfile)

# It loads the IP file and creates four dictionaries.
# The first one gives the devices of every ip, the second one the cookies of every ip, the third one the ips of every device and the last one the ips of every cookie.
(IPDev,IPCoo,DeviceIPS,CookieIPS)=loadIPS(ipfile,DictDevice,DictCookie,XIPS,Groups)


#################################################################################################
# PROCEDURE WITH THE INITIAL SELECTION OF CANDIDATES (PROCEDURE DESCRIBED IN THE DOCUMENTATION) #
#################################################################################################
print('STEP: Initial selection of candidates')
# Using simple rules with select a set of candidate cookies for every device
CandidatesTR=selectCandidates(DevicesTrain,Cookies,IPDev,IPCoo,DeviceIPS,CookieIPS,DictHandle)
CandidatesTST=selectCandidates(DevicesTest,Cookies,IPDev,IPCoo,DeviceIPS,CookieIPS,DictHandle)


#####################################################
# CREATION OF THE TRAINING AND TEST SET             #
# (THE FEATURES ARE DESCRIBED IN THE DOCUMENTATION) #
#####################################################

print('STEP: Creating the dataset')

# It creates the training and test set for supervised learning creating pairs (device,cookie) using the selected candidates.
(XTR,OriginalIndexTR)=createDataSet(CandidatesTR,DevicesTrain,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties)
YTR=createTrainingLabels(CandidatesTR,Labels)
(XTST,OriginalIndexTST)=createDataSet(CandidatesTST,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties)

######################################
# TRAINING USING BAGGING AND XGBOOST #
######################################

print('STEP: Training Supervised Learning')
(resultadosVal,resultadosTST, OriginalIndexTR,OriginalIndexTST, classifiers)=FullTraining(YTR,XTR,XTST,OriginalIndexTR,OriginalIndexTST,DevicesTrain, Groups, Labels)

###############################
# SAVING THE SUPERVISED MODEL #
###############################

print('Saving Supervised model')
with open(modelpath+'Model'+'.pkl', 'wb') as f:
    pickle.dump(classifiers, f)

#############################################################################
# UPDATING THE DATA STRUCTURES DATASETS WITH NEW INFORMATION OF THE RESULTS #
#############################################################################

print('Updating features for with semisupervised learning information')
# SECOND LOOP FOR SEMISUPERVISED LEARNING
# It repeats the training procedure adding the cookies with high probability to the WhosDevice structure
uniqueCand=uniqueCandidates(DevicesTest,Cookies,IPCoo,DeviceIPS,DictHandle,Groups)
probCand=mostProbable(resultadosTST, OriginalIndexTST, Groups)
DictOtherDevices=createOtherDevicesDict(Labels,uniqueCand,probCand)

# It creates the training and test set for supervised learning creating pairs (device,cookie) using the selected candidates.
(XTR,OriginalIndexTR)=createDataSet(CandidatesTR,DevicesTrain,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,DictOtherDevices,DevProperties)
(XTST,OriginalIndexTST)=createDataSet(CandidatesTST,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,DictOtherDevices,DevProperties)

####################################################################################################
# SECOND TRAINING USING XGBOOST AND BAGGNG INCLUDING THE NEW INFORMATION(SEMI SUPERVUSED LEARNING) #
####################################################################################################

# Training, it trains using 10 fold CV and the predicions are the average of the classifiers of every fold.
print('STEP: Training Semi-Supervised Learning')
(resultadosVal,resultadosTST, OriginalIndexTR,OriginalIndexTST, classifiers)=FullTraining(YTR,XTR,XTST,OriginalIndexTR,OriginalIndexTST,DevicesTrain, Groups, Labels)

############################################################
# POST PROCESSING PROCEDURE DESCRIBED IN THE DOCUMENTATION #
############################################################

print('STEP: Post Processing')
# Initial selection of the cookies associated to every device
(validat,thTR)=bestSelection(resultadosVal, OriginalIndexTR, np.array([1.0,0.9]),Groups)

# Increasing the number of candidates in devices whose best candiate doesn't have a good likelihood
(validat,thTR) = PostAnalysisTrain(validat,thTR,classifiers,DevicesTrain,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties,DictHandle,Labels)

F05=calculateF05(validat,Labels)
print "F05 Validation",F05

###################################
# SAVING THE SEMISUPERVISED MODEL #
###################################

print('Saving semisupervised model')
saveModel(modelpath1,classifiers,DictOtherDevices)
