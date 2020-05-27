import glob
import pandas as pd
import nltk
import re
import numpy as np
import os
import pickle

import datetime as dt
import gensim
import torch
import spacy
import neuralcoref
# import tensorflow as tf
# import tensorflow_hub as hub

from collections import namedtuple
from copy import deepcopy
from tqdm import tqdm
from functools import reduce
from autocorrect import Speller

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import genesis, wordnet_ic, wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn import metrics
from scipy import spatial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Globals
tokenizer = RegexpTokenizer(r'\w+') #will remove punctuations
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
genesis_ic=wn.ic(genesis, False, 0.0)
spell = Speller()
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)

def word_removal(sentence, character_length = 0):
    words = word_tokenize(sentence)
    words = [word for word in words if ((len(word) > character_length) or (re.match('[a-zA-Z]+',word) is None))]
    return ' '.join(words)

def get_testing_data(data_path = os.path.join(os.getcwd(),'Resource','Semeval_Data_XMLtocsv')):
    print('Loading sample data')
    # Load students' answers
    direc=glob.glob(os.path.join(data_path,'*.csv'))
    direc.sort()
    df_students_ans = []
    stud_ans = []
    accuracy = []
    accuracy_dict = {'incorrect':0,'correct':1}
    for s in direc:
        df_question_ans = pd.read_csv(s)
        question_answers = df_question_ans.__text.tolist()
        question_answers = [spell(ans.lower()) for ans in question_answers]
        stud_ans.append(question_answers)
        accuracy.append([accuracy_dict[acc] for acc in df_question_ans._accuracy.tolist()])
        df_students_ans.append(df_question_ans)
    df_students_ans = pd.concat(df_students_ans)

    # Load Reference answers
    ref_ans=[]
    with open(os.path.join(data_path,'ref_ans.txt'), 'r') as infile:
        for x in infile:
            if x=='\n':
                continue
            ref_ans.append(spell(x.strip().lower()))
    
    # Load Questions
    question=[]
    with open(os.path.join(data_path,'question.txt'), 'r') as infile:
        for x in infile:
            if x=='\n':
                continue
            question.append(spell(x.strip().lower()))
    
    total_student_responses = df_students_ans.shape[0]
    assert len(stud_ans) == len(ref_ans) == len(question), "Number of rows of answers, questions and reference answers must be equal"
    print(f'Total Responses: {total_student_responses}. Total Questions: {len(stud_ans)}')
    return stud_ans, ref_ans, question, total_student_responses, accuracy

def lsa_score(stud_ans, n_components = 3):
    '''
    Computes the first 3 eigenvalues for for all student answers, including prepended
    '''
    corpus = [ans for ques in stud_ans for ans in ques]
    vectorizer = TfidfVectorizer(min_df = 1, stop_words = 'english')
    dtm = vectorizer.fit_transform(corpus).astype(np.float32)
    lsa = TruncatedSVD(n_components, algorithm = 'arpack')
    dtm_lsa = lsa.fit_transform(dtm)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)
    lsa_stud = pd.DataFrame(dtm_lsa, index = corpus, columns = [f'component_{i+1}' for i in range(n_components)])
    return lsa_stud.values.tolist()

def word_sent_length(stud_ans):
    '''
    Computes, for each answer, the total number of valid characters that are not whitespaces divided by the total length of the sentence
    '''
    sentence_length=[0] * len(stud_ans)
    av_word_length=[0] * len(stud_ans)
    j=0
    for i in stud_ans:
        sentence_length[j]=len(tokenizer.tokenize(i))
        for character in i:
            if(character!=' '):
                av_word_length[j]+=1
        av_word_length[j]/=sentence_length[j]
        j+=1

    ws = [av_word_length, sentence_length]
    return ws

def prepare_word2vec(model_path = os.path.join('Resource','enwiki_20180420_100d.txt')):
    '''
    Prepare word2vec model
    '''
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    return model

def prepare_fasttext(model_path = os.path.join('Resource','cc.en.300.bin')):
    '''
    Prepare word2vec model
    '''
    model = gensim.models.fasttext.load_facebook_vectors(model_path, encoding='utf-8')
    return model

def prepare_roberta():
    '''
    Returns a roberta model for textual entailment
    '''
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')
    if torch.cuda.is_available():
        roberta.cuda()
    roberta.eval()  # disable dropout (or leave in train mode to finetune)
    return roberta

def cosine_sim_word2vec(stud_ans,ref_ans, w2vmodel):
    '''
    For each student answer, return the average cosine similarity between all the words in student answerand all the words in the reference (except stopwords)
    '''
    nums=[]
    for i in range(len(stud_ans)):
        ss1 = stud_ans[i]
        ss2 = ref_ans

        
        s1 = [w.lower() for w in tokenizer.tokenize(ss1) if w.lower() not in stop_words]
        s2 = [w.lower() for w in tokenizer.tokenize(ss2) if w.lower() not in stop_words]

        sim=0
        for i in s1:
            maxi=0
            for j in s2:
                maxi = max(maxi,w2vmodel.similarity(i,j))
            sim+=maxi

        length = max(len(word_tokenize(ss1)), len(word_tokenize(ss2)))
        sim/=length
        nums.append(sim)
    return nums

def prepare_doc2vec(answers):
    '''
    Return a matrix containing a representative vector for every student and teacher's answers
    '''
    document_corpus = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, text in enumerate(answers):
        words = text.lower().split()
        tags = [i]
        document_corpus.append(analyzedDocument(words, tags))

    # Train model (set min_count = 1, if you want the model to work with the provided example data set)
    model = gensim.models.Doc2Vec(document_corpus, vector_size = 48, window = 300, min_count = 1, workers = 4, epochs = 40)
    return model.docvecs

def cosine_sim_d2v(student_answers,reference_answers):
    '''
    Return the cosine similarity for every student answer and that of the reference answer, based on vectors computed from doc2vec model
    '''
    answers = reduce(lambda x,y: x+y,student_answers)
    total_answers = len(answers)
    answers += reference_answers
    matrix = prepare_doc2vec(answers)

    results = []
    std_ans_idx = np.cumsum([0] + [len(ans) for ans in student_answers])
    for idx, idx_val in enumerate(std_ans_idx):
        if idx > 0:
            for i in range(std_ans_idx[idx-1],idx_val):
                results.append(1 - spatial.distance.cosine(matrix[i], matrix[std_ans_idx[-1]+idx-1]))
    
    assert len(results) == total_answers
    return results

def average_sentence_length(stud_ans):
    '''
    Return average sentence length of all the words
    '''
    _, sentence_lengths = word_sent_length(stud_ans)
    return np.mean(sentence_lengths)

def prepare_IDF_info(student_answers):
    '''
    For each answer, return the frequency of words counted and the length of the answer in words
    '''
    doc_info = []
    for ans_idx, student_answer in enumerate(student_answers):
        tokenized_sentence = tokenizer.tokenize(student_answer)
        doc_info.append({"doc_id":ans_idx + 1, "doc_length(count)":len(tokenized_sentence)})

    word_freq = []
    for student_answer in student_answers:
        tokenized_sentence = tokenizer.tokenize(student_answer)
        word_histogram = {}
        for word in tokenized_sentence:
            word = word.lower()
            if word in word_histogram:
                word_histogram[word]+=1
            else:
                word_histogram[word]=1
        word_freq.append({'doc_id':student_answer, "freq":word_histogram})
    #print(freq,"\n\n\n\n\n")
    return doc_info, word_freq

#Tfidf vectorizer function
def calculate_IDF_score(student_answers):
    '''
    Return for each term in each document its inverse document frequency score
    '''
    doc_info, word_freq = prepare_IDF_info(student_answers)
    IDFscore=[]

    for counter, answer in enumerate(word_freq):
        for word in answer['freq'].keys():
            documents_containing_word = sum([word in document_word_frequency['freq'] for document_word_frequency in word_freq])
            IDFscore.append({'doc_id':counter+1, 'IDFscore':np.log(len(doc_info)/documents_containing_word),'TF score':(documents_containing_word),'key':word})

    return IDFscore

def finite_state_transducer_score(student_answers,reference_answer, w2vmodel, k1=1.2, b=0.75):
    '''
    Return finite state transducer score for all the student's answers
    '''
    fst_scores=[]
    avsenlength = average_sentence_length(student_answers)
    #compare stud_ans and ref_ans
    idfscore = calculate_IDF_score(student_answers)
    for student_answer in student_answers:
        # Get the character length of the shorter sentence in terms of words when comparing the student and the reference answer
        if(len(word_tokenize(student_answer)) > len(word_tokenize(reference_answer))):
            longer_sentence = [word.lower() for word in tokenizer.tokenize(student_answer)]
            shorter_sentence = [word.lower() for word in tokenizer.tokenize(reference_answer)]
            shorter_length = len(reference_answer)
        else:
            shorter_sentence = [word.lower() for word in tokenizer.tokenize(student_answer)]
            longer_sentence = [word.lower() for word in tokenizer.tokenize(reference_answer)]
            shorter_length = len(student_answer)
        
        fst_score = 0
        for word_ls in longer_sentence:
            maxi=0
            idf=0
            for word_ss in shorter_sentence:
                maxi = max(maxi,w2vmodel.similarity(word_ls,word_ss))

            for j in range(len(idfscore)):
                if(idfscore[j]['key'] == word_ls):
                    idf = idfscore[j]['IDFscore']

            fst_score += idf * (maxi * (k1+1))
            fst_score /= (maxi + k1* (1- b + b*(shorter_length/avsenlength)))
        fst_scores.append(fst_score)
    return fst_scores

def get_func_words(func_words_path = os.path.join(os.getcwd(),'Resource','function_words')):
    '''
    Returns a list of functional words by extracting it via regex from a file
    '''
    func_words=[]
    pattern=r"\w+'?\w*"
    with open(func_words_path) as infile:
        for i in infile.readlines():
            result = re.findall(pattern,i)
            for word in result:
                func_words.append(word)
    return func_words

# def content_overlap(student_answers,reference_answer,functional_words):
#     '''
#     Compute the overlap percentage for each student's answers, based on the lemmatised token
#     Denominator = number of lemamtised tokens excluding functional words in the reference answer
#     Using wordnet, obtain all synonyms for lemmatised tokens in the reference answer

#     Resolves the following:
#     1. Identical
#     2. Leammatisation
#     3. Spelling (handled beforehand)
#     4. Semantic similarity - using wordnet thesaurus
#     '''
#     student_answers_tokenized = []
#     reference_answer = deepcopy(reference_answer)
#     for student_answer in student_answers:
#         student_answers_tokenized.append(tokenizer.tokenize(student_answer))
#     for idx, student_ans_token in enumerate(student_answers_tokenized):
#         student_answers_tokenized[idx] = [lemmatizer.lemmatize(token.lower()) for token in student_ans_token if token not in functional_words]
#     reference_answer = [lemmatizer.lemmatize(i.lower()) for i in tokenizer.tokenize(reference_answer) if i not in functional_words]
#     length_reference = len(reference_answer)
#     for i in range(len(reference_answer)):
#         for j in wn.synsets(reference_answer[i]):
#             for k in j.lemmas():
#                 reference_answer.append(k.name())
    
#     overlap_percentage = []
#     for student_answer in student_answers_tokenized:
#         val = 0
#         for token in student_answer:
#             if token in reference_answer:
#                 val += 1
#         overlap_percentage.append(val/length_reference)
#     return overlap_percentage

def content_overlap(student_answers,reference_answer,functional_words, n_grams = [1]):
    '''
    Compute the overlap percentage for each student's answers, based on the lemmatised token
    Denominator = number of lemamtised tokens excluding functional words in the reference answer
    Using wordnet, obtain all synonyms for lemmatised tokens in the reference answer

    Can configure n-gram overlap

    Resolves the following:
    1. Identical
    2. Leammatisation
    3. Spelling (handled beforehand)
    4. Semantic similarity - using wordnet thesaurus

    If the answer is less than length of n-gram then the features will return 0
    '''

    student_answers_tokenized = []
    reference_answer = deepcopy(reference_answer)

    for student_answer in student_answers:
        student_answers_tokenized.append([token.lemma_ for token in nlp(student_answer) if not token.is_punct and not token.is_stop and token.lemma_ not in functional_words])

    # Reference answer is a list of lists. Append all synonyms to the list
    reference_answer = [[token.lemma_] for token in nlp(reference_answer) if not token.is_punct and not token.is_stop and token.lemma_ not in functional_words]
    for i in range(len(reference_answer)):
        for j in wn.synsets(reference_answer[i][0]):
            for k in j.lemmas():
                reference_answer[i].append(k.name())
    results = []
    for n_gram in n_grams:
        overlap_percentage = []
        for student_idx, student_answer in enumerate(student_answers_tokenized):
            overlap_count = 0
            for i in range(len(student_answer)):
                if i == len(student_answer)-n_gram+1:
                    break
                gram = student_answer[i:i+1+(n_gram-1)]
                for j in range(0,len(reference_answer)-n_gram+1):
                    if all([word in reference_answer[j+k] for k, word in enumerate(gram)]):
                        overlap_count += 1
            overlap_percentage.append(overlap_count/max(1,len(reference_answer)-n_gram+1))
        results.append(overlap_percentage)
    return results

def mnli_roberta(student_answers, reference_answer, roberta_model):
    '''
    returns an array for the MNLI RoBerTa prediction. Does a MNLI textual entailment comparison between every sentence in the answer and every sentence in the reference
    Sums the arrays up for every answer along the first axis, then flattens it
    '''
    roberta_model.eval()
    mnli_results = []
    for answer in student_answers:
        answer = nlp(answer)._.coref_resolved
        answer_array = []
        for ans_sentence in sent_tokenize(answer):
            for ref_sentence in sent_tokenize(reference_answer): #re.sub(r'\.\s*$','',reference_answer).split('.'):
                tokens = roberta_model.encode(ans_sentence, ref_sentence)
                answer_array.append(roberta_model.predict('mnli', tokens).detach().cpu().numpy())
        answer_array = np.array(answer_array).sum(axis = 0).flatten()
        mnli_results.append(answer_array)
    return mnli_results

def jclc_sim(student_answers,reference_answer):
    '''
    Returns 2 semantic similarity measures: namely Leacock-Chodorow and Jiang-Conrath
    Uses nltk Wordnet as the thesaurus graph
    Returns the Jiang-Conrath & leacock-chodorow Similarity Score based on the Information Content (IC) of the Least Common Subsumer (most specific ancestor node) and that of the two input Synsets.
    '''
    jc_results = []
    lc_results = []
    ref_words = tokenizer.tokenize(reference_answer)
    ref_words = [lemmatizer.lemmatize(j) for j in ref_words if j.lower() not in stop_words]
    for answer in student_answers:
        lc_num = 0
        jc_num = 0
        ans_words = tokenizer.tokenize(answer)
        ans_words = [lemmatizer.lemmatize(j) for j in ans_words if j.lower() not in stop_words]
        max_token_num = max(len(ref_words),len(ans_words))
        for word in ans_words:
            jc_maxi = 0
            lc_maxi = 0
            for w1 in wn.synsets(word):
                for ref_token in ref_words:
                    for w2 in wn.synsets(ref_token):
                        if w1._pos in ('n','v','a','r') and w2._pos in ('n','v','a','r') and w1._pos==w2._pos:
                            # Compute JC similarity
                            jc = w1.jcn_similarity(w2,genesis_ic)
                            if w1 == w2 or jc>1:
                                jc_maxi=1
                            else:
                                jc_maxi = max(jc_maxi,jc)
                            # Compute LC similarity
                            lc = w1.lch_similarity(w2,genesis_ic)
                            #print(w1, w2, type(n), n)
                            #return None when there is no similarity hence needed to add another if clause
                            if lc == None:
                                lc_maxi=0
                            elif w1==w2 or lc > 1:
                                lc_maxi=1
                            else:
                                lc_maxi = max(lc_maxi,lc)
            jc_num=jc_num+jc_maxi
            lc_num=lc_num+lc_maxi
        jc_num=jc_num/max_token_num
        lc_num=lc_num/max_token_num
        jc_results.append(jc_num)
        lc_results.append(lc_num)
    return jc_results, lc_results

def chunk_overlap(student_answers,reference_answer, patterns = None):
    '''
    Computes the chunk overlap between the reference and student answer.
    Denominator is the total number of unique chunks in the reference answer
    Specify an optiomal item, otherwise the default will be used

    Chunks may consist of only 1 word

    NOTE: No functional words are excluded here
    '''

    ref_answer = nlp(reference_answer)
    if patterns is None:
        patterns = [
                [{'POS': 'ADJ'}],[{'POS': 'ADJ', 'OP': '?'},
                {'POS': 'NOUN', 'OP': '+'}],
                ]

    matcher = spacy.matcher.Matcher(nlp.vocab)
    for i, pat in enumerate(patterns):
        matcher.add(f'P{i}', None, pat)
        
    ref_matches = matcher(ref_answer)
    ref_array = np.zeros(len(ref_answer))
    for match_id, start, end in ref_matches:
        ref_array[start:end] = 1
        
    ref_matches_no = []
    match = []
    for i in range(len(ref_answer)):
        if ref_array[i] == 1:
            match.append(ref_answer[i])
        elif (i > 0 and ref_array[i] == 0 and ref_array[i-1] ==1):
            ref_matches_no.append(match)
            match = []
        elif i == len(ref_answer)-1 and len(match) > 0:
            ref_matches_no.append(match)

    ref_matches_no = set([' '.join([str(t.lemma_) for t in m]) for m in ref_matches_no])
    results = []
    for student_answer in student_answers:
        student_overlap = 0
        student_answer = nlp(student_answer)
        student_answer = ' '.join([re.sub('\s','',token.lemma_) for token in student_answer if not token.is_punct]) # Remove all empty spaces, if the token is an empty space
        for chunk in ref_matches_no:
            if chunk in student_answer:
                student_overlap += 1
        results.append(student_overlap/len(ref_matches_no))

    return results

# def embed_sentence_use(answers, module_url = os.path.join(os.getcwd(),'Resource','uni_encoder')):
#     '''
#     Return embeddings for the universal sentence encoder
#     '''
#     tf.logging.set_verbosity(tf.logging.ERROR)
#     #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
#     embed = hub.Module(module_url)
#     with tf.Session() as session:
#         session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#         embeddings = session.run(embed(answers))
#     return embeddings

# def cosine_sim_use(student_answers,reference_answer):
#     '''
#     Return cosine similarity using embeddings of the sentence in the universal sentence encoder
#     https://arxiv.org/abs/1803.11175
#     Only available in TF
#     '''
#     answers = student_answers[:]
#     answers.append(reference_answer)
#     matrix = embed_sentence_use(answers)
#     cossim_use=[0] * (len(matrix)-1)
#     for i in range(len(matrix)-1):
#         cossim_use[i] = 1 - spatial.distance.cosine(matrix[i], matrix[len(matrix)-1])
#     return cossim_use

def prompt_overlap(student_answers,question):
    '''
    Computes overlap between the student answer and the question
    '''
    qs_words = tokenizer.tokenize(question)
    q_words = [w for w in qs_words if w not in stop_words]
    results = []
    for answer in student_answers:
        overlap_metric = 0
        w = tokenizer.tokenize(answer)
        for j in w:
            for k in q_words:
                if(j==k):
                    overlap_metric+=1
                    break
        overlap_metric /= len(q_words)
        results.append(overlap_metric)
    return results


def get_features(student_answers, reference_answers, questions, w2v_model, functional_words, roberta_model):
    '''
    Student answers should be a list of list of answers for each question
    Reference answers should be a list of answers, 1 answer per question
    Questions is a lit of questions
    '''
    # Compute Feature Scores 
    f123_lsa_scores = lsa_score(student_answers)
    f456_content_overlap = []
    f7_cs_word2vec = []
    f8_cs_doc2vec = cosine_sim_d2v(student_answers, reference_answers) # Build the doc2vec model
    f9_fsts = []
    f10_roberta_mnli = []
    f11_12_jclc_sim = []
    f13_prompt_overlap = []
    f14_chunk_overlap = []

    # Iterate through all rows
    with tqdm(enumerate(zip(student_answers, reference_answers, questions)), desc = 'Generating Features' ,total = len(student_answers)) as iterator:
        for _, (stud_ans, ref_ans, question) in iterator:
            # time = dt.datetime.now()
            f456_content_overlap.append(content_overlap(stud_ans, ref_ans, functional_words, n_grams = [1,2,3]))
            # print('Word Sent Time',dt.datetime.now()-time)
            # time = dt.datetime.now()

            f7_cs_word2vec.append(cosine_sim_word2vec(stud_ans, ref_ans, w2v_model))
            # print('W2V cs Time',dt.datetime.now()-time)
            # time = dt.datetime.now()

            f9_fsts.append(finite_state_transducer_score(stud_ans, ref_ans, w2v_model))
            # print('TF-IDF Time',dt.datetime.now()-time)
            # time = dt.datetime.now()

            f10_roberta_mnli.append(mnli_roberta(stud_ans, ref_ans, roberta_model))
            # print('RoBerTa',dt.datetime.now()-time)
            # time = dt.datetime.now()

            f11_12_jclc_sim.append(jclc_sim(stud_ans, ref_ans))
            # print('JCLC Sim',dt.datetime.now()-time)
            # time = dt.datetime.now()

            f13_prompt_overlap.append(prompt_overlap(stud_ans, question))
            # print('Prompt Overlap',dt.datetime.now()-time)
            # time = dt.datetime.now()
            f14_chunk_overlap.append(chunk_overlap(stud_ans, ref_ans))

    # Post Processing
    f123_lsa_scores = np.array(f123_lsa_scores).T.tolist()
    f10_roberta_mnli = np.concatenate([np.array(f) for f in f10_roberta_mnli]).T.tolist()
    
    features = [
        # 1-3. LSA Components 1,2,3
        f123_lsa_scores[0],
        f123_lsa_scores[1],
        f123_lsa_scores[2],
        # 4. Unigram Overlap
        reduce(lambda x,y: x+y,[f[0] for f in f456_content_overlap]),
        # 5. Bigram Overlap
        reduce(lambda x,y: x+y,[f[1] for f in f456_content_overlap]),
        # 6. Trigram overlap
        reduce(lambda x,y: x+y,[f[2] for f in f456_content_overlap]),

        # 6. Word 2 Vec Cosine Similarity
        reduce(lambda x,y: x+y,f7_cs_word2vec),
        # 7. Doc 2 Vec Cosine Similarity
        f8_cs_doc2vec,
        # 8. TF-IF Scores
        reduce(lambda x,y: x+y,f9_fsts),
        # 10 RoBerTa
        f10_roberta_mnli[0],
        f10_roberta_mnli[1],
        f10_roberta_mnli[2],
        # 11 JC sim
        reduce(lambda x,y: x+y,[f[0] for f in f11_12_jclc_sim]),
        # 12 LC sim
        reduce(lambda x,y: x+y,[f[1] for f in f11_12_jclc_sim]),
        # 13 Prompt Overlap
        reduce(lambda x,y: x+y,f13_prompt_overlap),
        # 13 Chunk Overlap
        reduce(lambda x,y: x+y,f14_chunk_overlap)
    ]

    features = np.array(features).T

    return features

def get_features_test_data(w2v_model, functional_words, roberta_model, sample_data = None, save_path = os.path.join(os.getcwd(),'Resource')):
    '''
    Returns the feature matrix from the test data
    '''
    if sample_data is None:
        student_answers, reference_answers, questions, total_responses, accuracy = get_testing_data()
    else:
        student_answers, reference_answers, questions, total_responses, accuracy = sample_data
    accuracy = np.array(reduce(lambda x,y: x+y,accuracy))
    
    features = get_features(student_answers, reference_answers, questions, w2v_model, functional_words, roberta_model)

    assert features.shape[0] == accuracy.shape[0], 'Feature Matrix and Y values do not have the same number of rows'
    # Save Feature
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    np.save(os.path.join(save_path,'X.npy'), features)
    np.save(os.path.join(save_path,'y.npy'), accuracy)
    
    return features, accuracy

def get_metrics(y_pred, y_test):
    train_metrics = dict(
        accuracy = metrics.accuracy_score(y_test, y_pred),
        f1 = metrics.f1_score(y_test, y_pred),
        precision = metrics.precision_score(y_test, y_pred),
        recall = metrics.recall_score(y_test, y_pred)
    )
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('F1: ', metrics.f1_score(y_test, y_pred))
    print('Precision: ', metrics.precision_score(y_test, y_pred))
    print('Recall: ', metrics.recall_score(y_test, y_pred))
    return train_metrics

def load_classifier(classifier_path = os.path.join(os.getcwd(),'Resource','classifier_model.pk')):
    classifier_model = pickle.load(open(classifier_path, 'rb'))
    return classifier_model

def train_test_model(X = None, y = None, w2v_model = None, functional_words = None, roberta_model = None, classifier_model = None, sample_data = None, save_path = os.path.join(os.getcwd(),'Resource'), tune_threshold_flag = True, threshold = 0.35, random_state = 1):
    '''
    Trains the classifier model based on the sample data and the 
    '''
    if X is None and (w2v_model is None or functional_words is None or roberta_model is None):
        raise Exception('If X is not provided, you have to provide the necessary models to generate the features')
    elif X is None:
        try:
            X = np.load(os.path.join(save_path,'X.npy'))
        except FileNotFoundError:
            X, y = get_features_test_data(w2v_model, functional_words, roberta_model, sample_data = sample_data)
            
            # Scale and save the mean and variances to be reused later
            # Only scale if X is newly computed
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            np.save(os.path.join(save_path,'X_mean.npy'), scaler.mean_)
            np.save(os.path.join(save_path,'X_var.npy'), scaler.var_)
    
    if y is None:
        try:
            y = np.load(os.path.join(save_path,'y.npy'))
        except FileNotFoundError:
            _, _, _, _, y = get_testing_data()
            y = np.array(reduce(lambda x,y: x+y,y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = random_state)

    if classifier_model is None:
        classifier_model = RandomForestClassifier(n_estimators = 300, max_depth = 15, random_state = random_state)

    classifier_model.fit(X_train, y_train)
    y_pred = classifier_model.predict_proba(X_test)

    if tune_threshold_flag:
        threshold = tune_threshold(y_pred, y_test)
        print(f'Tuned Threshold: {threshold:.2f}')

    y_pred = (y_pred[:,1] > threshold).astype(np.long)
    metrics = get_metrics(y_pred, y_test)

    pickle.dump(classifier_model, open(os.path.join(save_path,'classifier_model.pk'), "wb" ))

    return classifier_model, metrics

def tune_threshold(y_pred, y_val, metric = metrics.accuracy_score, divs = 25):
    '''
    Gives the best threshold by grid search
    '''
    thresholds = np.linspace(0,1,divs).tolist()
    metric_values = []
    for t in thresholds:
        y_pred_long = (y_pred[:,1] > t).astype(np.long)
        metric_values.append(metric(y_val, y_pred_long))
    return thresholds[metric_values.index(max(metric_values))]
    

def get_probabilities(student_answers, reference_answers, questions, w2v_model, functional_words, roberta_model, classifier_model, y_truth = None, answer_features = None, save_path = os.path.join(os.getcwd(),'Resource')):
    '''
    Feed in GIM student answers, reference answers, questions and models to get the probability of a correct score
    '''
    # Sense check
    assert len(student_answers) == len(reference_answers) == len(questions), 'Student answers, reference answers, and questions must have the same length'
    student_answers, reference_answers, questions = deepcopy(student_answers), deepcopy(reference_answers), deepcopy(questions)
    
    # Spell and convert to lowercase
    for idx, question_ans in enumerate(student_answers):
        student_answers[idx] = [spell(ans.lower()) for ans in question_ans]
        reference_answers[idx] = spell(reference_answers[idx].lower())
        questions[idx] = spell(questions[idx].lower())

    if answer_features is None:
        X = get_features(student_answers, reference_answers, questions, w2v_model, functional_words, roberta_model)
    else:
        X = answer_features

    # Scale using sample dataset features
    X_mean = np.load(os.path.join(save_path,'X_mean.npy'))
    X_var = np.load(os.path.join(save_path,'X_var.npy'))
    X = (X-X_mean)/np.sqrt(X_var)

    y_pred = classifier_model.predict_proba(X)

    if y_truth is not None:
        threshold = tune_threshold(y_pred, y_truth)
        print(f'Tuned Threshold: {threshold:.2f}')
    
    return X, y_pred