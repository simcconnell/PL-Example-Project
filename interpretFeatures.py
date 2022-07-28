"""
This file contains instructions for dealing with feature names as stored in data files.

buildInterpreter: builds dictionary to translate feature abbreviations into names
getMatchFeatures: get list of features, excluding bookmaker numbers

The following are helper functions to complete the csv file containing the feature dictionary.
    findUnknownAbbreviations, printUnknownAbbreviations (from data files)
    findUnusedAbbreviations, printUnusedAbbreviations (from feature dictionary)
    addUnknownsToDictionary (update feature dictionary)
"""

import csv


def buildInterpreter(csvfile):
    with open(csvfile) as csvfile:
        featureReader = csv.reader(csvfile, delimiter='=')
        featureDict = dict()
        for row in featureReader:
            featureAbbreviation = row[0].strip()
            featureName = row[1].strip()
            featureDict[featureAbbreviation] = featureName
        return featureDict


def getMatchFeatures(csvfile, numMatchFeatures):
    with open(csvfile) as csvfile:
        featureReader = csv.reader(csvfile, delimiter='=')
        rowCounter = 0
        matchFeatures = [''] * numMatchFeatures
        for row in featureReader:
            featureName = row[0].strip()
            matchFeatures[rowCounter] = featureName
            rowCounter += 1
            if rowCounter >= numMatchFeatures:
                break
    return matchFeatures


def findUnknownAbbreviations(csvfile, featureDict):
    with open(csvfile) as csvfile:
        featureReader = csv.DictReader(csvfile)
        header = featureReader.fieldnames
        unknowns = []
        for abbrev in header:
            if abbrev not in featureDict:
                unknowns.append(abbrev)
    return unknowns


def printUnknownAbbreviations(csvfile, featureDict):
    unknowns = findUnknownAbbreviations(csvfile, featureDict)
    unknowns.sort()
    print("The following feature abbreviations are unknown:")
    print(unknowns)


def findUnusedAbbreviations(csvfile, featureDict):
    with open(csvfile) as csvfile:
        featureReader = csv.DictReader(csvfile)
        header = featureReader.fieldnames
        unused = []
        for abbrev in featureDict:
            if abbrev not in header:
                unused.append(abbrev)
    return unused


def printUnusedAbbreviations(csvfile, featureDict):
    unused = findUnusedAbbreviations(csvfile, featureDict)
    unused.sort()
    print("The following feature abbreviations are unused:")
    print(unused)


def addUnknownsToDictionary(featureFile, unknowns):
    # pass in featureDictionary.csv and list of unknown abbreviations
    # add rows of the form abbrev, abbrev (unknown abbreviation)
    with open(featureFile, mode='a') as csvfile:
        writer = csv.writer(csvfile, delimiter='=', lineterminator='\n')
        newRows = [[]] * len(unknowns)
        for i in range(len(unknowns)):
            abbrev = unknowns[i]
            newRows[i] = [abbrev, abbrev + ' (unknown abbreviation)']
        writer.writerows(newRows)
