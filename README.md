# FAQ-matching
## Pipeline
1. Question or not question binary classifier (~99.82% accuracy, trained on SQuAD 2.0 dataset and SPAADIA dataset)

        Adapted from https://github.com/lettergram/sentence-classification

2. Ensemble model to score the similarity between 2 texts (i.e. FAQ and user's query)


        Test results based on 50 relevant (in FAQ) and 42 irrelevant questions:
        TP = 31; FP = 19; FN = 6; TN = 36

## Installation Guide
Run the setup.sh file