import pandas as pd

import math
import numpy as np
import scipy.stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Graphics Libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine Learning Libraries
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

# Import data handling libraries
import datetime as dt

def helloWorld():
  print("Hello, World!")

def loadAndCleanData(filename):
    data = pd.read_csv(filename)
    data.fillna(0)
    #print(data)
    return data

def computeProbability(feature, bin, data):
    # Count the number of datapoints in the bin
    count = 0.0

    for i,datapoint in data.iterrows():
        # See if the data is in the right bin
        if datapoint[feature] >= bin[0] and datapoint[feature] < bin[1]:
            count += 1

    # Count the total number of datapoints
    totalData = len(data)

    # Divide the number of people in the bin by the total number of people
    probability = count / totalData

    # Return the result
    return probability

def computeConfidenceInterval(data):
      # Confidence intervals
      npArray = 1.0 * np.array(data)
      stdErr = scipy.stats.sem(npArray)
      n = len(data)
      return stdErr * scipy.stats.t.ppf((1+.95)/2.0, n - 1)

def getEffectSize(d1,d2):
    m1 = d1.mean()
    m2 = d2.mean()
    s1 = d1.std()
    s2 = d2.std()

    return (m1 - m2) / math.sqrt((math.pow(s1, 3) + math.pow(s2, 3)) / 2.0)

def runTTest(d1,d2):
    return scipy.stats.ttest_ind(d1,d2)

# pip install statsmodels
# vars is a string with our independent and dependent variables
# " dvs ~ ivs"
def runANOVA(dataframe, vars):
    model = ols(vars, data=dataframe).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    return aov_table

# Plot a timeline of my data
def plotTimeline(data, time_col, val_col):
    sns.lineplot(data=data, x=time_col, y=val_col)
    plt.show()

# Plot a timeline of my data broken down by each category (cat_col)
def plotMultipleTimelines(data, time_col, val_col, cat_col):
    sns.lineplot(data=data, x=time_col, y=val_col, hue=cat_col)
    plt.show()

# Run a linear regression over the data. Models an equation
# as y = mx + b and returns the list [m, b].
def runTemporalLinearRegression(data, x, y):
    # Format our data for sklean by reshaping from columns to np arrays
    x_col = data[x].map(dt.datetime.toordinal).values.reshape(-1,1)
    y_col = data[y].values.reshape(-1, 1)

    # Run the regression using an sklearn regression object
    regr = LinearRegression()
    regr.fit(x_col, y_col)

    # Compute the R2 score and print it. Good scores are close to 1
    y_hat = regr.predict(x_col)
    fitScore = r2_score(y_col, y_hat)
    print("Linear Regression Fit: " + str(fitScore))

    # Plot linear regression against data. This will let us visually judge whether
    # or not our model is any good. With small data, a high R2 doesn't always mean
    # a good model: we can use our intuition as well.
    plt.scatter(data[x], y_col, color='lightblue')
    plt.plot(data[x], y_hat, color='red', linewidth=2)
    plt.show()

    # y = mx + b
    # Return m and b
    return [regr.coef_[0][0], regr.intercept_[0]]


# Define a logistic function that we can use to model logistic data without
# requiring classification.
def logistic(x, x0, m, b):
    y = 1.0 / (1.0 + np.exp(-m*(x - x0) + b))
    return (y)

# Define a logistic modeling regression. Use this regression for modeling the
# data rather than a classification. Note that your y value must be between
# 0 and 1 for this function to work correctly.
def runTemporalLogisticRegression(data, x, y):
    # Process the data
    x_col = data[x].map(dt.datetime.toordinal)
    y_col = data[y]

    # Give the curve a crappy fit to start with
    # In this case, we'll start with x0 as the median and define a straight
    # line between 0 and 1. The curve_fit function will adjust the line
    # to minimize the residuals.
    p0 = [np.median(x_col), 1, min(y_col)]
    params, pcov = curve_fit(logistic, x_col, y_col, p0)

    # Show the fit with the actual data in blue and the model in red. Note that
    # m = params[1] and b = params[2].
    plt.scatter(data[x], y_col, color='lightblue')
    plt.plot(data[x], logistic(x_col, params[0], params[1], params[2]), color='red', linewidth=2)
    plt.show()

    return params
