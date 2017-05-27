
import csv
import os
import re
import numpy as np
import inspect
import sys
import sklearn
from sklearn import cross_validation
import pickle


from Variables import *
from VariablesTST import *

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


# This part loads the content of the devices into a numpy matrix using the dictionaries to transform the text values into numerical values
print('Loading Devices Files')
DevicesTrain = loadDevices(trainfile,DictHandle,DictDevice,DictDevType,DictDevOs,DictCountry,DictAnnC1,DictAnnC2)

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


#########################
# LOADING THE TEST FILE #
#########################

print('STEP: Loading test file')
DevicesTest = loadDevices(predictFile,DictHandle,DictDevice,DictDevType,DictDevOs,DictCountry,DictAnnC1,DictAnnC2)
(LabelsTest,Groups,WhosDevice)=creatingLabels(DevicesTest,Cookies,DictHandle)

###################################
# INITIAL SELECTION OF CANDIDATES #
###################################

print('STEP: Initial selection of candidates')
CandidatesTST=selectCandidates(DevicesTest,Cookies,IPDev,IPCoo,DeviceIPS,CookieIPS,DictHandle)

#####################
# LOADING THE MODEL #
#####################

print('Loading the model') 
(classifiers,DictOtherDevices) = loadModel(modelpath1)

########################
# CREATING THE DATASET #
########################

print('STEP: Creating the dataset')
(XTST,OriginalIndexTST)=createDataSet(CandidatesTST,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,DictOtherDevices,DevProperties)

########################
# USING THE CLASSIFIER #
########################

print('STEP: Using the classifier')
resultadosTST = Predict(XTST,classifiers)

########################
# POST PROCESSING STEP #
########################

print('STEP: Post Processing')
(validatTST,thTST)=bestSelection(resultadosTST, OriginalIndexTST, np.array([1.0,0.9]),Groups)

(validatTST,thTST) = PostAnalysisTest(validatTST,thTST,classifiers,DevicesTest,Cookies,DeviceIPS,CookieIPS,IPDev,IPCoo,Groups,WhosDevice,DevProperties,DictHandle)

F05=calculateF05(validatTST,LabelsTest)
print "F05 Validation",F05

#########################################
# WRITIG THE FINAL SOLUTION IN THE FILE #
#########################################

print('Writing the file with the rvalidatTSTesult')
writeSolution(resultFile,validatTST,DeviceList,CookieList)
