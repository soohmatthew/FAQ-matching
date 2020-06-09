from features.misc import nltk_init
nltk_init()
from model import ASAG
from question_classification.model import *

import os
import numpy as np
import pandas as pd
import re
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import json
import string

def load_faq():
    with open("FAQ/QNA.json") as f:
        faq = json.load(f)

    return faq

# Inter quartile range
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    upper_bound = quartile_3 + (iqr * 2)
    return np.where(ys > upper_bound)

def get_question(event):
    faq = load_faq()
    questions = [list(faq.keys())]
    query = [event["query"]]
    one_ans = asag.grade(questions, query, [''], y_truth = None)
    if len(outliers_iqr(one_ans[:,1])[0]) == 1:
        question = questions[0][outliers_iqr(one_ans[:,1])[0][0]]       
        return {'question' : question, 'answer' : faq[question]}
    else:
        return {'question' : 'NA', 'answer' : 'NA'}

if __name__ == '__main__':
    asag = ASAG()
    asag.load_train_model()
    qc = Question_classifier()
    qc.load_model()

    with open('FAQ/test_FAQ.txt', 'r') as file:
        data = file.read()

    questions = []
    predicted_question = []
    predicted_answer = []
    is_question = []
    for q in data.split('\n'):
        questions.append(q)
        is_question.append(qc.predict(q))
        response = get_question({'query': q})
        predicted_question.append(response['question'])
        predicted_answer.append(response['answer'])

    test_set = pd.DataFrame.from_dict({'questions': questions, 'is_question' : is_question, 'predicted_question' : predicted_question, 'predicted_answer' : predicted_answer})

    ## TO-DO: output confusion matrix
    test_set.to_csv('test_set_results.csv')
    print('Saved test set results')