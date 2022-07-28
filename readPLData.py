"""
This file contains instructions for reading and cleaning Premier League data.
The data is contained in csv files for each year of the league's existence.
While the column names do overlap, they are not identical. Older data is much sparser.

I've excluded information from bookmakers.

fileFromNumber: get file name from season number
intervalFileList: create list of file names for an (inclusive) interval of season numbers
getTeams: get list of teams which have played in PL and the seasons in which they've played
findNonNumericColumns: finds non-numeric columns for potential cleaning
getRefereeList: lists all referees, including redundancies
sortSurname: sorts list by longest part of name, which is usually the surname
mergeDuplicateReferees: write csv mapping aliases of a given name to preferred format

The following are just helper functions:
    findRefereeDuplicates
    askHowManyNames
    getTrueNames
    findAliases
    getWholeRefereeName
    getRefereeSurname
"""

import pandas as pd
import string


def fileFromNumber(seasonNumber):
    if seasonNumber < 10:
        addZero = '0'
    else:
        addZero = ''
    fileName = 'plData/pl' + addZero + str(seasonNumber) + '.csv'
    return fileName


def intervalFileList(first, last):
    fileList = []
    for season in range(first, last + 1):
        fileName = fileFromNumber(season)
        fileList.append(fileName)
    return fileList


def getTeams(seasonNumbers):
    teams = dict()
    for season in seasonNumbers:
        seasonDF = pd.read_csv(fileFromNumber(season))
        colNames = seasonDF.head()
        if 'HomeTeam' in colNames:
            for teamName in seasonDF['HomeTeam']:
                if teamName in teams:
                    teams[teamName].add(season)
                else:
                    teams[teamName] = {season}
        else:
            print('No home team column in file for season ' + str(season) + '.')
    return teams


def findNonNumericColumns(startSeason, endSeason, featureList, numTypes):
    messyColumnList = set()
    for seasonNumber in range(startSeason, endSeason + 1):
        seasonDF = pd.read_csv(fileFromNumber(seasonNumber), usecols=featureList)
        seasonDF.info(show_counts=True, memory_usage=False, verbose=False)

        nonNumericColumns = seasonDF.select_dtypes(exclude=numTypes)
        nonNumericColumnHeads = set(nonNumericColumns.columns)
        messyColumnList = messyColumnList.union(nonNumericColumnHeads)

    if len(messyColumnList) == 0:
        print("All columns are numeric.")
    else:
        print("All non-numeric columns:")
        print(messyColumnList)

    return messyColumnList


def getRefereeList(startSeason, endSeason):
    # List all referees by surname to detect duplicates
    refereeList = set()
    for seasonNumber in range(startSeason, endSeason + 1):
        seasonDF = pd.read_csv(fileFromNumber(seasonNumber), usecols=['Referee'])
        seasonDF['Referee'] = seasonDF['Referee'].str.strip()
        refereeList = refereeList.union(seasonDF['Referee'])
    return sortSurname(list(refereeList))


def sortSurname(nameList):
    # sort list (mostly) by surname, given names written in varying formats
    # actually sorts by longest portion of name
    nameList = sorted(nameList, key=lambda x: max(x.split(), key=len))
    return nameList


def mergeDuplicateReferees(refList, refDict, fileName):
    # given sorted list of referees, merge different formats of the same name
    # returns dictionary mapping each name to preferred format, writes to csv
    startIdx = 0

    while startIdx < len(refList):
        curWholeName = getWholeRefereeName(refList, startIdx)
        curSurname = getRefereeSurname(curWholeName)
        endIdx = startIdx + 1
        while endIdx < len(refList) and getRefereeSurname(getWholeRefereeName(refList, endIdx)) == curSurname:
            endIdx += 1
        if endIdx - startIdx == 1:
            refDict[curWholeName] = curWholeName
        else:
            findRefereeDuplicates(refList, refDict, startIdx, endIdx)
        startIdx = endIdx

    print("Here is the thinned list of referees.")
    uniqueReferees = list(set(refDict.values()))
    uniqueReferees = sortSurname(uniqueReferees)
    for ref in uniqueReferees:
        print(ref)
    print("Any other duplicates should be removed manually.")
    with open(fileName, 'w') as csvfile:
        csvfile.write('Alias,True Name\n')
        for alias in refDict.keys():
            csvfile.write("%s;%s\n" % (alias, refDict[alias]))
    return refDict


def findRefereeDuplicates(refList, refDict, startIdx, endIdx):
    # get user input to determine which referee names are redundant
    print("Here is a list of possibly redundant names:")
    for i in range(startIdx, endIdx):
        print(str(i - startIdx + 1) + ") " + refList[i])

    numTrueNames = askHowManyNames()

    # if duplicates, ask user to identify preferred version of name
    if numTrueNames < endIdx - startIdx:
        trueNamesIndices = getTrueNames(numTrueNames, endIdx - startIdx)
        nameMap = {i: [] for i in trueNamesIndices}
        if numTrueNames > 1:
            for i in trueNamesIndices:
                nameMap[i] = findAliases(refList, startIdx + i - 1, endIdx - startIdx)
        else:
            nameMap[trueNamesIndices[0]] = range(1, endIdx - startIdx + 1)
        for i in trueNamesIndices:
            for j in nameMap[i]:
                refDict[refList[startIdx + j - 1]] = refList[startIdx + i - 1]
    else:
        for i in range(startIdx, endIdx):
            refDict[refList[i]] = refList[i]


def askHowManyNames():
    while True:
        try:
            numTrueNames = int(input("How many distinct names are listed above?\n"))
        except ValueError:
            print("Input must be an integer.")
        else:
            if numTrueNames < 0:
                print("Input cannot be negative.")
            else:
                return numTrueNames


def getTrueNames(numTrueNames, numCandidates):
    # get indices of names listed in preferred formats
    shortPluralString = "s"
    longPluralString = "s, separated by semicolons"
    if numTrueNames == 1:
        shortPluralString = ""
        longPluralString = ""
    message = "Enter the number" + shortPluralString + " corresponding to %d" % numTrueNames
    message = message + " correctly formatted name" + longPluralString + ".\n"
    while True:
        try:
            nameIndices = list(map(int, input(message).split(';')))
        except ValueError:
            print("Inputs must be integers.\n")
        else:
            if len(nameIndices) != numTrueNames:
                print("Wrong number of inputs.")
            elif min(nameIndices) < 1 or max(nameIndices) > numCandidates:
                print("Inputs must lie between 1 and %d.\n" % numCandidates)
            else:
                return nameIndices


def findAliases(refList, trueIdx, numCandidates):
    # get indices of aliases of a particular name
    while True:
        try:
            message = "Please enter the numbers corresponding to aliases of %s, " \
                      "separated by semicolons.\n" % refList[trueIdx]
            aliasIndices = list(map(int, input(message).split(';')))
        except ValueError:
            print("Inputs must be integers.")
        else:
            if min(aliasIndices) < 1 or max(aliasIndices) > numCandidates:
                print("Inputs must be between 1 and %d.\n" % numCandidates)
            else:
                break
    return aliasIndices


def getWholeRefereeName(refList, i):
    return refList[i].translate(str.maketrans('', '', string.punctuation)).strip()


def getRefereeSurname(name):
    return max(name.split(), key=len)
