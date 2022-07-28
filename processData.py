"""
This file contains functions for converting raw csv files to usable dataframes.

processSeason: read in data for a particular season
convertToNumeric: convert all non-numeric columns in dataframe using ColumnTransformer
featureLabelSplitNames: get new feature and label names after running columnTransformer
featureLabelSplitData: split data into features and labels
checkContinue: check with user to continue or exit
checkDataMerge: verify that datasets merged as expected

The following are just helper functions:
    readRefFile
    translateRefereeColumn
    parseDateInfo
"""

import csv
import datetime
import numpy as np
import pandas as pd
import sys

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import readPLData


def processSeason(seasonNumber, refDict, dictEmpty, fileName, features):
    seasonDF = pd.read_csv(readPLData.fileFromNumber(seasonNumber),
                           usecols=features,
                           parse_dates=['Date'],
                           infer_datetime_format=True, dayfirst=True)
    seasonDF = translateRefereeColumn(seasonDF, refDict, dictEmpty, fileName)
    seasonDF = parseDateInfo(seasonDF, seasonNumber)
    return seasonDF


def convertToNumeric(train, test, nonNumerics, toNumStr):
    ct = ColumnTransformer([(toNumStr, OneHotEncoder(), nonNumerics)],
                           remainder='passthrough')

    trainSize = train.shape[0]
    testSize = test.shape[0]
    allData = pd.concat([train, test], ignore_index=True, copy=False)
    allData = pd.DataFrame(ct.fit_transform(allData).toarray(), columns=ct.get_feature_names_out())

    return allData.loc[0:trainSize], allData.loc[trainSize:trainSize+testSize]


def featureLabelSplitNames(allCols, oldLabels, prefix):
    # get new feature and label names after running columnTransformer
    newLabelNames = []
    for label in oldLabels:
        newLabel = prefix + '__' + label
        newLabelNames.append(newLabel)
    newFeatureNames = [feat for feat in allCols if feat not in newLabelNames]
    return newFeatureNames, newLabelNames


def featureLabelSplitData(allData, features, labels, normalize):
    dataFeatures = allData[features]
    dataLabels = allData[labels[0]] - allData[labels[1]]
    if normalize:
        dataLabels = dataLabels.apply(np.sign)
    return dataFeatures, dataLabels


def checkContinue():
    while True:
        userAnswer = input("Continue program? (Y/N)\n").lower()
        if userAnswer == 'y':
            return
        elif userAnswer == 'n':
            sys.exit("Exiting program.")
        else:
            print("Invalid response.")


def checkDataMerge(data1, data2):
    numNaN = data1.isna().sum().sum() + data2.isna().sum().sum()
    if numNaN > 0:
        print("There are %d NaN entries in the merged dataset. You should look into this." % numNaN)
        sys.exit("Exiting program.")
    else:
        currentTime = datetime.datetime.now()
        timeString = currentTime.strftime("%H:%M:%S")
        print("Successfully read in data to produce %d samples and %d test cases (%s)."
              % (data1.shape[0], data2.shape[0], timeString))


def readRefFile(refDict, filename):
    # convert ref csv file to dict
    with open(filename, newline='') as csvFile:
        refReader = csv.reader(csvFile, delimiter=';')
        next(refReader)
        for row in refReader:
            alias = row[0]
            trueName = row[1]
            refDict[alias] = trueName


def translateRefereeColumn(df, refDict, dictEmpty, filename):
    # in 'Referee' column, replace aliases with true names
    if dictEmpty:
        readRefFile(refDict, filename)
    df['Referee'] = df['Referee'].str.strip()
    return df.replace(refDict)


def parseDateInfo(df, seasonNumber):
    # given dataframe for a season, add column indicating season number and replace dates with months
    seasonCol = [seasonNumber] * df.shape[0]
    df['Season'] = seasonCol
    df['Date'] = pd.DatetimeIndex(df['Date']).month
    return df
