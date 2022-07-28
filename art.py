"""
This file contains functions for plotting various pieces of data.

createFeatureTimeline: create figure displaying match features available in each year to select usable features
createHistoryPlot: plot accuracy over number of preceding seasons used to predict
plotResults: display information about error and estimator coefficients

The following are just helper functions:
    drawFeatureTimeline
    plotListOfStacks
    plotStack
    initHistoryPlot
    initResultsPlot
    initDataFromDict
    initDataFromList
"""


import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import readPLData

DEFAULT_COLOR = 'b'
ALT_COLOR = 'r'
DOT_SIZE = 6


def createFeatureTimeline(featureList, startSeason, endSeason):
    # create figure displaying match features available in each year
    # used to create USABLE_FEATURES list and FIRST_USEFUL_SEASON
    dataHeaders = dict()
    for seasonNumber in range(startSeason, endSeason + 1):
        seasonDF = pd.read_csv(readPLData.fileFromNumber(seasonNumber))
        seasonFeatures = seasonDF.head()
        dataHeaders[seasonNumber] = seasonFeatures
    drawFeatureTimeline(featureList, range(startSeason, endSeason + 1), dataHeaders)


def createHistoryPlot(title, yVal, numXVals, stacks):
    figure, axis = initHistoryPlot(title, yVal, numXVals)
    for k, y in stacks.items():
        plotStack(axis, k, y)
    plt.show()


def plotResults(errorDict, homeDict, awayDict, refBiasDict, statDict):
    seasonList, errorList = initDataFromList(errorDict)
    teamList, homeList = initDataFromDict(homeDict)
    awayList = [list(map(lambda x: -x, awayDict[teamList[j]])) for j in range(len(teamList))]
    refList, refBiasList = initDataFromDict(refBiasDict)
    statList, statSignificanceList = initDataFromDict(statDict)

    fig, axes = initResultsPlot(seasonList, teamList, refList, statList)
    axes[0, 0].plot(seasonList, errorList)
    plotListOfStacks(axes[0, 1], homeList, DEFAULT_COLOR)
    plotListOfStacks(axes[0, 1], awayList, ALT_COLOR)
    plotListOfStacks(axes[1, 0], statSignificanceList, DEFAULT_COLOR)
    plotListOfStacks(axes[1, 1], refBiasList, DEFAULT_COLOR)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.95, hspace=0.5)
    plt.show()


def drawFeatureTimeline(featureList, seasonList, dataHeaders):
    # plot features available in each season, in order to select usable features
    # dataHeaders is dict from season number to header of corresponding data file
    plt.xlim(0, len(seasonList) + 1)
    plt.ylim(0, len(featureList) + 1)
    plt.grid()
    plt.yticks([i + 1 for i in range(len(featureList))], featureList)

    for i in range(len(featureList)):
        feature = featureList[i]
        x = [j for j in seasonList if feature in dataHeaders[j]]
        y = [i + 1] * len(x)
        plt.plot(x, y, marker="o", markersize=6)
    plt.show()


def plotListOfStacks(axis, stackList, c):
    for i in range(len(stackList)):
        plotStack(axis, i, stackList[i], c)


def plotStack(axis, x, yList, color=DEFAULT_COLOR):
    xList = [x] * len(yList)
    axis.scatter(xList, yList, marker='.', c=color, s=DOT_SIZE)
    axis.scatter([x], [sum(yList) / len(yList)], marker='*', c=color)


def initHistoryPlot(plotTitle, yTitle, maxX):
    fig, ax = plt.subplots()
    fig.suptitle(plotTitle + ' Accuracy')
    ax.set_xlabel('History Length')
    ax.set_ylabel(yTitle)
    ax.set_xticks(range(0, maxX))

    return fig, ax


def initResultsPlot(seasons, teams, refs, stats):
    f, a = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [3, 7]})
    f.suptitle('Results', fontweight='bold')

    a[0, 0].set(title='Mean-Squared Error', xticks=seasons, xlabel='Season')

    a[0, 1].set(title='Team Results')
    a[0, 1].set_xticks(range(len(teams)), teams, rotation='vertical')
    homeListPatch = mlines.Line2D([], [], color=DEFAULT_COLOR, marker='.', linestyle='None',
                                  label='Home win coefficients')
    homeAveragePatch = mlines.Line2D([], [], color=DEFAULT_COLOR, marker='*', linestyle='None',
                                     label='Average home win coefficient')
    awayListPatch = mlines.Line2D([], [], color=ALT_COLOR, marker='.', linestyle='None',
                                  label='Away win coefficients')
    awayAveragePatch = mlines.Line2D([], [], color=ALT_COLOR, marker='*', linestyle='None',
                                     label='Average away win coefficient')
    a[0, 1].legend(handles=[homeListPatch, homeAveragePatch, awayListPatch, awayAveragePatch])

    a[1, 0].set(title='Other Statistics')
    a[1, 0].set_xticks(range(len(stats)), stats, rotation='vertical')
    statListPatch = mlines.Line2D([], [], color=DEFAULT_COLOR, marker='.', linestyle='None', label='Coefficients')
    statAveragePatch = mlines.Line2D([], [], color=DEFAULT_COLOR, marker='*', linestyle='None', label='Average')
    a[1, 0].legend(handles=[statListPatch, statAveragePatch])

    a[1, 1].set(title='Referee Bias Toward Home Team')
    a[1, 1].set_xticks(range(len(refs)), refs, rotation='vertical')
    refListPatch = mlines.Line2D([], [], color=DEFAULT_COLOR, marker='.', linestyle='None', label='Coefficients')
    refAveragePatch = mlines.Line2D([], [], color=DEFAULT_COLOR, marker='*', linestyle='None', label='Average')
    a[1, 1].legend(handles=[refListPatch, refAveragePatch])

    return f, a


def initDataFromDict(dictionary):
    # given dictionary mapping keys to lists, sort by average of list
    zippedList = [(k, v) for k, v in sorted(dictionary.items(), key=lambda item: sum(item[1])/len(item[1]))]
    keyList, valueList = zip(*zippedList)
    return list(keyList), list(valueList)


def initDataFromList(data):
    tuple1, tuple2 = zip(*data)
    return list(tuple1), list(tuple2)
