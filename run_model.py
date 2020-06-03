from model import ASAG
import os
import numpy as np
import pandas as pd
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import features.functional as F

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    asag = ASAG()
    asag.load_train_model()