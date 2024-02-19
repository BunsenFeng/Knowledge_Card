# retrieval-augmented factuality filter, one component in the knowledge card framework

import transformers
import torch
import numpy as np
import wikipedia as wp
import sys
import time

vitaminc_model = None
factkb_model = None

def load_factuality_model(devices):
    global vitaminc_model, factkb_model
    tokenizer1 = transformers.AutoTokenizer.from_pretrained("tals/albert-xlarge-vitaminc-mnli")
    tokenizer2 = transformers.AutoTokenizer.from_pretrained("roberta-base")
    vitaminc_model = transformers.pipeline('text-classification', model="tals/albert-xlarge-vitaminc-mnli", tokenizer = tokenizer1, device = devices[0], return_all_scores = True, max_length = 512, truncation = True)
    factkb_model = transformers.pipeline('text-classification', model="bunsenfeng/FactKB", tokenizer = tokenizer2, device = devices[1], return_all_scores = True, max_length = 512, truncation = True)

def factuality(text):
    docs = [""]
    # while True:
    #     try:
    #         docs = docs + [wp.summary(ent, sentences=5) for ent in wp.search(text, results = 3)]
    #         break
    #     except:
    #         print("wiki search limit exceeded, retrying...")
    #         time.sleep(10)
    try:
        for ent in wp.search(text[:100], results = 3):
            try:
                docs.append(wp.summary(ent, sentences=5))
            except:
                # print("error in retrieving summary for " + ent)
                pass
    except:
        print("error in wiki search")
        time.sleep(2)
        pass
    #docs = [wp.summary(ent, sentences=10) for ent in wp.search(text, results = 5)]
    global vitaminc_model, factkb_model
    if vitaminc_model is None or factkb_model is None:
        raise Exception("factuality model not loaded")
    scores = []
    # for doc in docs:
    #     text_post = text + " " + doc
    #     vitaminc_score = vitaminc_model(text_post)
    #     factkb_score = factkb_model(text_post)
    #     print(vitaminc_score)
    #     print(factkb_score)
    #     exit(0)
    text_posts = [text + " " + doc for doc in docs]
    vitaminc_scores = vitaminc_model(text_posts)
    factkb_scores = factkb_model(text_posts)
    for i in range(len(docs)):
        vitaminc_score = (vitaminc_scores[i][0]['score'] - vitaminc_scores[i][1]['score'] + 0 * vitaminc_scores[i][2]['score'] + 1) / 2 # 0 to 1
        factkb_score = factkb_scores[i][1]['score'] # 0 to 1
        scores.append((vitaminc_score + factkb_score) / 2)
    return np.max(scores)

# sort the list of texts by factuality and return top k
def factuality_filter(texts, k = 3):
    global vitaminc_model, factkb_model
    if vitaminc_model is None or factkb_model is None:
        raise Exception("factuality model not loaded")
    scores = []
    for i in range(len(texts)):
        try:
            scores.append((i, factuality(texts[i])))
        except:
            print("factuality score calc error")
            scores.append((i, 0.5))

    # retain top 2*k scores, redistribute the probability for sampling
    scores.sort(key = lambda x: x[1], reverse = True)
    scores = scores[:2 * k]
    scores = [(scores[i][0], scores[i][1] / np.sum([scores[j][1] for j in range(len(scores))])) for i in range(len(scores))]
    # sample k texts with probability scores
    indices = np.random.choice([scores[i][0] for i in range(len(scores))], k, p = [scores[i][1] for i in range(len(scores))])
    return [texts[int(indices[i])] for i in range(k)]

    # # normalize scores with softmax
    # scores = np.array(scores)
    # scores[:, 1] = np.exp(scores[:, 1])
    # scores[:, 1] = scores[:, 1] / np.sum(scores[:, 1])
    # # sample k texts with probability scores
    # indices = np.random.choice(scores[:, 0], k, p = scores[:, 1])
    # return [texts[int(indices[i])] for i in range(k)]

    # scores.sort(key = lambda x: x[1], reverse = True)
    # return [texts[scores[i][0]] for i in range(k)]

# test
# if __name__ == "__main__":
#     load_factuality_model([0, 1])
#     text = ["The capital of the United States is Washington, D.C.", "I am stupid", "I am a genius", "i am a", "a a a", "b b b", "c c c", "d d d"]
#     print(factuality_filter(text))