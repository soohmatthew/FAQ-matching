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
import logging
from time import gmtime, strftime

def load_faq():
    with open("FAQ/QNA.json") as f:
        faq = json.load(f)

    return faq

# Inter quartile range
def outliers_iqr(ys, threshold = 2):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    upper_bound = quartile_3 + (iqr * threshold)
    return np.where(ys > upper_bound)

def get_question(event):
    faq = load_faq()
    questions = [list(faq.keys())]
    query = [event["query"]]
    one_ans = asag.grade(questions, query, [''], y_truth = None)
    if len(outliers_iqr(one_ans[:,1])[0]) == 1:
        question = questions[0][outliers_iqr(ys = one_ans[:,1], threshold = event['threshold'])[0][0]]       
        return {'question' : question, 'answer' : faq[question]}
    else:
        return {'question' : 'NA', 'answer' : 'NA'}

if __name__ == '__main__':
    thresholds = [1.5,2, 2.5, 3]
    params = {'lsa' : True, 
                'content_overlap' : True,
                'w2v' : True, 
                'd2v' : True, 
                'fsts' : True,
                'roberta' : True,
                'jclc' : True,
                'chunk_overlap' : True}
    asag = ASAG(params = params)
    asag.load_train_model()
    #logging.critical('Similarity Model initialized')
    qc = Question_classifier()
    qc.load_model()
    #logging.critical('Question Classifier initialized')
    
    for threshold in thresholds:
        if not os.path.exists('logs'):
            os.mkdir('logs')
        logging.basicConfig(filename = f"logs/test_set_log_{threshold}_{strftime('%Y-%m-%d_%H_%M_%S', gmtime())}.log", format = '%(asctime)s %(message)s',level=logging.CRITICAL)
        logging.critical(f"Threshold = {threshold}")
        logging.critical(f'{params}')

        with open('FAQ/test_FAQ.txt', 'r') as file:
            data = file.read()

        questions = data.split('\n')
        predicted_question = []
        predicted_answer = []
        is_question = []
        for q in data.split('\n'):
            is_question.append(qc.predict(q))
            response = get_question({'query': q, 'threshold' : threshold})
            predicted_question.append(response['question'])
            predicted_answer.append(response['answer'])

        test_set = pd.DataFrame.from_dict({'questions': questions, 'is_question' : is_question, 'predicted_question' : predicted_question, 'predicted_answer' : predicted_answer})
        
        # Question Classifier test
        ACC = sum(is_question)/len(is_question)
        logging.critical(f"Question Classifier Accuracy: {ACC}")

        # FAQ similarity test, first 50 are relevant, rest are irrelevant
        FN = sum([1 for q in predicted_question[:51] if q == 'NA'])
        TP = 50 - FN
        TN = sum([1 for q in predicted_question[51:] if q == 'NA'])
        FP = len(predicted_question[51:]) - TN

        logging.critical(f"FAQ Similarity FN: {FN}")
        logging.critical(f"FAQ Similarity TP: {TP}")
        logging.critical(f"FAQ Similarity TN: {TN}")
        logging.critical(f"FAQ Similarity FP: {FP}")
        if not os.path.exists('test_results'):
            os.mkdir('test_results')
        test_set.to_csv(f'test_results/test_set_results_{threshold}.csv')
        print('Saved test set results')