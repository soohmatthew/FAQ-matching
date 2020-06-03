import os
import features.functional as F

from features.misc import log_print

class ASAG(object):
    def __init__(self,
                classifier_model = None,
                classifier_threshold = 0.5,
                logger = None,
                random_state = 1
                ):
        self.classifier_threshold = classifier_threshold
        self.logger = logger
        self.classifier_model = classifier_model
        self.paths = dict(

        )
        self.loaded = dict(
            models_loaded = False,
            sample_data_loaded = False,
            classifier_loaded = False
        )
        self.random_state = random_state

    def load_models(self, word_model = 'fasttext'):
        if word_model == 'fasttext':
            self.word_model = F.prepare_fasttext()
        elif word_model == 'word2vec':
            self.word_model = F.prepare_word2vec()
        else:
            raise NotImplementedError(f'{word_model} is not implemented')
        log_print('Models - Word Model Loaded', self.logger)

        self.roberta = F.prepare_roberta()
        log_print('Models - RoBERTa Loaded', self.logger)
        self.functional_words = F.get_func_words()
        log_print('Models - Functional Words Loaded', self.logger)

        self.loaded['models_loaded'] = True
        log_print('Models - Loading Complete', self.logger)

    def load_sample_data(self):
        self.train_data = F.get_train_data()
        self.test_data = F.get_test_data()
        self.loaded['sample_data_loaded'] = True
        log_print('Sample Data Loaded', self.logger)
    
    def load_classifier(self):
        self.classifier_model = F.load_classifier()
        self.loaded['classifier_loaded'] = True
        log_print('Classifier Loaded', self.logger)

    def load_all(self):
        self.load_models()
        self.load_sample_data()
        self.load_classifier()
        log_print('Loading Complete', self.logger)
    
    def load_train_model(self):
        self.load_models()
        self.load_sample_data()
        self.train_model()

    def train_model(self, features_selected = None):
        if not self.loaded['models_loaded'] or not self.loaded['sample_data_loaded']:
            raise Exception('Load Text Models and Sample Data First')

        self.features_selected = features_selected

        self.classifier_model, self.sample_data_metrics = F.train_test_model(train_data = self.train_data,
                                                test_data = self.test_data,
                                                w2v_model = self.word_model,
                                                functional_words = self.functional_words,
                                                roberta_model = self.roberta,
                                                threshold = self.classifier_threshold,
                                                classifier_model = self.classifier_model if self.classifier_model is not None else None,
                                                random_state = self.random_state,
                                                features_selected = features_selected
                                                )
        self.loaded['classifier_loaded'] = True
        log_print('Model Trained and Stored', self.logger)
    
    def grade(self, student_answers, reference_answers, questions, y_truth = None, answer_features = None):
        if not all(self.loaded.values()):
            raise Exception('Load Text Models, Sample Data and Classifier First')
        X, y_pred = F.get_probabilities(student_answers, reference_answers, questions, self.word_model, self.functional_words, self.roberta, self.classifier_model, y_truth = y_truth, answer_features = answer_features, features_selected = self.features_selected)
        self.answer_features = X
        return y_pred