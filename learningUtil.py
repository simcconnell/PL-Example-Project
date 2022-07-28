"""
This file contains machine learning methods.

neuralNetworkPipeline: run grid search with multilayer perceptron classifier and scaler
svmPipeline: run grid search with epsilon-support vector regression and scaler
recordResults: save error and estimator coefficients for svm to csv
readResults: read csv file of results, produce dictionaries of coefficients
createLayerList: create list of possible layer arrangements for neural network
interpretGoal: get function parameters for the sake of classification or regression

The following are just helper functions:
    addCoefficient
    printPipelineDetails
    iterateLayers
    addToDictList

"""

import csv
import itertools
import sys

from sklearn import metrics
from sklearn import model_selection
from sklearn import neural_network
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import svm


def neuralNetworkPipeline(X_train, Y_train, X_test, Y_test, layers, actFns, alphas, crossVal, verbose):
    pipe = pipeline.Pipeline([('sc', preprocessing.StandardScaler()), ('nn', neural_network.MLPClassifier())])
    params = {'nn__hidden_layer_sizes': layers,
              'nn__activation': actFns,
              'nn__alpha': alphas,
              'sc__copy': [False],
              'sc__with_mean': [False]}
    search = model_selection.GridSearchCV(estimator=pipe, param_grid=params, cv=crossVal, n_jobs=-1)
    search.fit(X_train, Y_train)

    if verbose:
        printPipelineDetails(search, X_test, Y_test)

    accuracy = metrics.accuracy_score(y_true=Y_test, y_pred=search.predict(X_test))

    return accuracy, -1, search.best_estimator_.named_steps['nn'].get_params()


def svmPipeline(X_train, Y_train, X_test, Y_test, kernels, polyDegrees, kTerms, regTerms, epTerms, crossVal, verbose):
    pipe = pipeline.Pipeline(steps=[('sc', preprocessing.StandardScaler()), ('svm', svm.SVR())])
    params = {'svm__kernel': kernels,
              'svm__degree': polyDegrees,
              'svm__coef0': kTerms,
              'svm__C': regTerms,
              'svm__epsilon': epTerms,
              'sc__copy': [False],
              'sc__with_mean': [False]}
    search = model_selection.GridSearchCV(estimator=pipe, param_grid=params, cv=crossVal, n_jobs=-1)
    search.fit(X_train, Y_train)

    if verbose:
        printPipelineDetails(search, X_test, Y_test)

    Y_predicted = search.predict(X_test)
    mse = metrics.mean_squared_error(Y_test, Y_predicted)

    return search.score(X_test, Y_test), mse, search.best_estimator_.named_steps['svm'].coef_


def recordResults(seasonNumber, mse, featureNames, featureCoefficients, fileName):
    with open(fileName, 'a') as csvfile:
        csvfile.write('Predictions for season %d\n' % seasonNumber)
        csvfile.write('Mean squared error: %f\n' % mse)
        for i in range(len(featureNames)):
            feature = featureNames[i]
            coefficient = featureCoefficients[0][i]
            csvfile.write("%s,%s\n" % (feature, coefficient))
        csvfile.write('\n')


def readResults(fileName, stringsToDelete):
    errors = []
    homeTeams = dict()
    awayTeams = dict()
    refs = dict()
    stats = dict()
    curSeason = 0

    with open(fileName) as csvfile:
        resultsReader = csv.reader(csvfile, delimiter=',')
        for row in resultsReader:
            if row:
                featureName = row[0]
                for i in range(len(stringsToDelete)):
                    featureName = featureName.replace(stringsToDelete[i], '')

                if featureName[:4] == 'Pred':
                    curSeason = int(featureName[23:])
                elif featureName[:4] == 'Mean':
                    errors.append((curSeason, float(featureName[20:])))
                else:
                    coefficient = float(row[1])
                    addCoefficient(featureName, coefficient, homeTeams, awayTeams, refs, stats)

    return errors, homeTeams, awayTeams, refs, stats


def createLayerList(minLayers, maxLayers, layerSizes):
    layerMem = dict()
    for k in range(minLayers, maxLayers + 1):
        iterateLayers(k, layerSizes, layerMem)
    return list(itertools.chain.from_iterable(layerMem.values()))


def interpretGoal(goalInput):
    if goalInput == 'Classification':
        return 'nn', 'Accuracy'
    elif goalInput == 'Regression':
        return 'svm', 'Score'
    else:
        sys.exit("Invalid goal.")


def addCoefficient(feature, coefficient, homeDict, awayDict, refDict, statDict):
    if feature.__contains__('HomeTeam'):
        feature = feature.replace('HomeTeam_', '')
        addToDictList(homeDict, feature, coefficient)
    elif feature.__contains__('AwayTeam'):
        feature = feature.replace('AwayTeam_', '')
        addToDictList(awayDict, feature, coefficient)
    elif feature.__contains__('Referee'):
        feature = feature.replace('Referee_', '')
        addToDictList(refDict, feature, coefficient)
    else:
        addToDictList(statDict, feature, coefficient)


def printPipelineDetails(estimator, testFeatures, testLabels):
    means = estimator.cv_results_['mean_test_score']
    stds = estimator.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, estimator.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

    print('Best parameters:')
    print(estimator.best_params_)

    print('Results on the test set:')
    print(metrics.classification_report(testLabels, estimator.predict(testFeatures)))


def iterateLayers(numLayers, sizes, mem):
    if numLayers not in mem:
        if numLayers == 1:
            layer = [tuple([x]) for x in sizes]
            mem[1] = layer
        else:
            iterateLayers(numLayers - 1, sizes, mem)
            prev = mem[numLayers - 1]
            oneLayer = mem[1]
            mem[numLayers] = [x + y for x in prev for y in oneLayer]


def addToDictList(dictionary, key, value):
    if key in dictionary:
        dictionary[key] = dictionary[key] + [value]
    else:
        dictionary[key] = [value]
    return dictionary
