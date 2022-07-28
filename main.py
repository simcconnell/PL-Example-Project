import pandas as pd

import art
import interpretFeatures
import learningUtil
import processData
import readPLData


# data files are labelled by season number, e.g. 01 for 1993-94 and 12 for 2004-05
FIRST_SEASON = 1
FIRST_USEFUL_SEASON = 8             # data is much sparser before this season
LAST_SEASON = 28

FEATURE_DICTIONARY_NAME = 'featureDictionary.csv'
REF_FILE_NAME = 'referees.csv'
RESULTS_FILE_NAME = 'results.csv'

NUMERIC_TYPES = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
NON_NUMERICS = ['HomeTeam', 'AwayTeam', 'HTR', 'Referee']           # removed 'FTR'
NUM_MATCH_FEATURES = 36
USABLE_FEATURES = ['Date',
                   'HomeTeam',
                   'AwayTeam',
                   'HTHG',
                   'HTAG',
                   'HTR',
                   'Referee',
                   'HS',
                   'AS',
                   'HST',
                   'AST',
                   'HC',
                   'AC',
                   'HF',
                   'AF',
                   'HY',
                   'AY',
                   'HR',
                   'AR']
OUTPUT_COLUMNS = ['FTHG', 'FTAG']                                   # removed 'FTR'
NUM_STR = 'toNumeric'
REM_STR = 'remainder'

CROSS_VALIDATION = 5
MAX_LAYERS = 4
LAYER_SIZES = [20, 50, 100]
HISTORY_LENGTH = 5


def runPreliminaryFunctions(features, refDict):
    # To be run before anything else and ignored once data has been sufficiently cleaned
    # Step 1: find features which are present in all recent seasons (results in selectFeatures)
    art.createFeatureTimeline(features, FIRST_SEASON, LAST_SEASON)
    print("At this point, you should check that the FIRST_USEFUL_SEASON and USABLE_FEATURES "
          "constants are consistent with the feature timeline.\n"
          "If they are not, update them now.")
    processData.checkContinue()

    # Step 2: check for duplicate teams (none found)
    allTeams = readPLData.getTeams(range(FIRST_USEFUL_SEASON, LAST_SEASON + 1)).keys()
    print(sorted(allTeams))
    print("At this point, you should check that there are no redundant team names.\n"
          "If there are, correct them before continuing.")
    processData.checkContinue()

    # Step 3: find non-numeric columns to clean (results in readPLDATA)
    nonNumericColumns = readPLData.findNonNumericColumns(FIRST_USEFUL_SEASON, LAST_SEASON, USABLE_FEATURES,
                                                         NUMERIC_TYPES)
    problemColumns = nonNumericColumns.difference(NON_NUMERICS)
    if len(problemColumns) == 0:
        print("This program already deals with all non-numeric columns.")
    else:
        print("At this point, you should exit the program to deal with the following non-numeric columns:")
        for col in problemColumns:
            print(col)
            print("These columns should be added to NON_NUMERICS or addressed manually.")
    processData.checkContinue()

    # Step 4: clean up referee list (results in referees.csv)
    refList = readPLData.getRefereeList(FIRST_USEFUL_SEASON, LAST_SEASON)
    refDict = readPLData.mergeDuplicateReferees(refList, refDict, REF_FILE_NAME)
    print("At this point, you should double check for redundancies in the list of referees.")
    processData.checkContinue()

    return refDict


def processAllData(seasonsToTrain, seasonsToTest, refDict, dictEmpty):
    # read files for a list of season numbers, merge, convert to usable format
    trainingList = [None] * len(seasonsToTrain)
    for i in range(len(seasonsToTrain)):
        trainingList[i] = processData.processSeason(seasonsToTrain[i], refDict, dictEmpty, REF_FILE_NAME,
                                                    USABLE_FEATURES + OUTPUT_COLUMNS)
    train = pd.concat(trainingList, ignore_index=True, copy=False)

    testingList = [None] * len(seasonsToTest)
    for i in range(len(seasonsToTest)):
        testingList[i] = processData.processSeason(seasonsToTest[i], refDict, dictEmpty, REF_FILE_NAME,
                                                   USABLE_FEATURES + OUTPUT_COLUMNS)
    test = pd.concat(testingList, ignore_index=True, copy=False)

    processData.checkDataMerge(train, test)

    return processData.convertToNumeric(train, test, NON_NUMERICS, NUM_STR)


def learnSeasons(learningType, seasonsToTrain, seasonsToTest, refDict, dictEmpty):
    classification = (learningType == 'nn')

    train, test = processAllData(seasonsToTrain, seasonsToTest, refDict, dictEmpty)
    newFeatNames, newLabelNames = processData.featureLabelSplitNames(list(train.columns), OUTPUT_COLUMNS, REM_STR)
    trainFeatures, trainLabels = processData.featureLabelSplitData(train, newFeatNames, newLabelNames, classification)
    testFeatures, testLabels = processData.featureLabelSplitData(test, newFeatNames, newLabelNames, classification)

    if learningType == 'nn':
        activationFns, layerList, alphaValues = initHyperparameters(learningType)
        return learningUtil.neuralNetworkPipeline(trainFeatures, trainLabels, testFeatures, testLabels, layerList,
                                                  activationFns, alphaValues, CROSS_VALIDATION, False)

    if learningType == 'svm':
        kernelFns, degrees, kernelTerms, regularizationTerms, tubeTerms = initHyperparameters(learningType)
        results = learningUtil.svmPipeline(trainFeatures, trainLabels, testFeatures, testLabels, kernelFns, degrees,
                                           kernelTerms, regularizationTerms, tubeTerms, CROSS_VALIDATION, False)
        return results + tuple([newFeatNames])


def initHyperparameters(MLType):
    if MLType == 'nn':
        # fns = ['identity', 'logistic', 'tanh', 'relu']
        fns = ['logistic']
        # layers = learningUtil.createLayerList(1, MAX_LAYERS, LAYER_SIZES)
        layers = learningUtil.createLayerList(2, 3, [50])
        # alphas = 10.0 ** -np.arange(1, 7)
        alphas = [10.0 ** -x for x in [2, 4, 6]]
        return fns, layers, alphas
    if MLType == 'svm':
        # fns = ['linear', 'poly', 'rbf', 'sigmoid']
        fns = ['linear']
        # degs = range(2,6)
        degs = [3]
        # kers = [10 ** -2, 1, 4]
        kers = [0.0]
        regs = [0.1, 5, 10]
        tubes = [10 ** -x for x in [0, 2, 4]]
        return fns, degs, kers, regs, tubes


def findBestHistoryLength(goal, refDict, dictEmpty):
    # for each k, predict each season using preceding k seasons, plot to find best k
    mlType, measuredValue = learningUtil.interpretGoal(goal)
    stacksToPlot = dict()

    for k in range(1, LAST_SEASON - FIRST_USEFUL_SEASON + 1):
        numCases = LAST_SEASON - k + 1 - FIRST_USEFUL_SEASON
        y = [0.0] * numCases
        for j in range(numCases):
            score, e, m, f = learnSeasons(mlType, range(FIRST_USEFUL_SEASON + j, FIRST_USEFUL_SEASON + j + k),
                                          [FIRST_USEFUL_SEASON + j + k], refDict, dictEmpty)
            y[j] = score
        stacksToPlot[k] = y

    art.createHistoryPlot(goal, measuredValue, LAST_SEASON - FIRST_USEFUL_SEASON + 1, stacksToPlot)


def computeResults(dictEmpty):
    numCases = LAST_SEASON - HISTORY_LENGTH + 1 - FIRST_USEFUL_SEASON
    for j in range(numCases):
        print("Started case %d of %d." % (j + 1, numCases))
        seasonToPredict = FIRST_USEFUL_SEASON + j + HISTORY_LENGTH
        s, e, m, f = learnSeasons('svm', range(FIRST_USEFUL_SEASON + j, FIRST_USEFUL_SEASON + j + HISTORY_LENGTH),
                                  [seasonToPredict], refereeDict, dictEmpty)
        learningUtil.recordResults(seasonToPredict, e, f, m, RESULTS_FILE_NAME)


if __name__ == '__main__':
    featureDict = interpretFeatures.buildInterpreter(FEATURE_DICTIONARY_NAME)
    matchFeatures = interpretFeatures.getMatchFeatures(FEATURE_DICTIONARY_NAME, NUM_MATCH_FEATURES)
    refereeDict = dict()

    runPreliminaryFunctions(matchFeatures, refereeDict)

    # findBestHistoryLength('Classification', refereeDict, True)
    findBestHistoryLength('Regression', refereeDict, True)

    computeResults(True)

    eResults, HTResults, ATResults, refResults, statsResults = learningUtil.readResults('results.csv',
                                                                                        [NUM_STR, REM_STR, '__'])
    art.plotResults(eResults, HTResults, ATResults, refResults, statsResults)
