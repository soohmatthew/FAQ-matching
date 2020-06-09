import json
import string

def load_faq():
    with open("QNA.json") as f:
        faq = json.load(f)

    return faq

def lambda_handler(event, context):
    faq = load_faq()
    questions = faq.keys()

    query = event["query"]
    high_score = 0
    argmax = None
    for qn in questions:
        score = similarity_score(query, qn)
        #print("qn: {} score: {}".format(qn, score))
        if score > high_score:
            high_score = score
            argmax = qn

    if argmax == None:
        return "No similar question in FAQ"

    return faq[argmax]

def similarity_score(sentence1, sentence2):
    words1 = preprocess(sentence1).split(" ")
    words2 = preprocess(sentence2).split(" ")
    intersection = len(set(words1) & set(words2))
    union = len(words1) + len(words2)
    return intersection / union

def preprocess(sentence):
    sentence = sentence.replace("?", "").replace(",", "")
    return sentence


if __name__ == "__main__":
    event = {"query": "How do I start investing?"}
    context = None
    #sentence1 = "What are the different forms companies submit and which do we get the information we want like quarterly report or insider ownership?"
    #sentence2 = "What forms do companies submit?"
    #print(similarity_score(sentence1, sentence2))
    print(lambda_handler(event, context))


