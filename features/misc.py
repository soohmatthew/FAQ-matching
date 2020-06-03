import os
import re
import nltk

def log_print(text, logger, log_only = False):
    if not log_only:
        print(text)
    if logger is not None:
        logger.info(text)

def nltk_init():
    nltk.download('wordnet')
    nltk.download('genesis')
    nltk.download('punkt')