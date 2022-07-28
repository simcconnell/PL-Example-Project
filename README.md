# PL-Example-Project

This repository exists only for the purpose of my current job search. Its goal is to demonstrate my coding style.

This project contains several files designed to take a cursory look at Premier League data. While it does contain a couple of methods using neural networks, it is centered primarily around sklearn.svm.SVR (with linear kernel, to predict goal differential). This is not the best way to accurately predict outcomes of matches, nor do I use features which are appropriate for predicting results. I selected this method because its output is relatively straightforward to interpret. I designed this project mostly to satisfy a mild curiosity about the correlation between various match statistics and the final outcome.

I used data available at https://www.football-data.co.uk/englandm.php.

File | Description
------|------
art.py | methods for creating png files
featureDictionary.csv | meanings of feature abbreviations, as given on website listed above
histLen.png | plot (k, accuracy of predicting after learning from preceding k seasons)
interpretFeatures.py | methods for dealing with features as stored in data files
learningUtil.py | methods for running NN/SVM, recording and reading results
main.py | high-level methods for reading and cleaning data, applying ML modules
matchFeaturesByYear.png | availability of features in each data file
processData.py | methods for converting raw csv files to usable dataframes
readPLData.py | methods for reading and cleaning Premier League data
referees.csv | referee names given in alternate formats
resultPlot.png | visual interpretation of results
results.csv | raw results
