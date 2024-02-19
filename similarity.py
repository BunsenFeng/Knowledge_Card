# similarity filter, one component in the knowledge card framework

import transformers
import torch
import numpy as np

sim_model = None

def load_similarity_model(device):
    global sim_model
    model_name_or_path = "microsoft/mpnet-base"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    sim_model = transformers.pipeline('feature-extraction', tokenizer = tokenizer, model=model_name_or_path, device = device, max_length = 200, truncation = True)

# def one_vector(text):
#     global sim_model
#     if sim_model is None:
#         raise Exception("similarity model not loaded")
#     return np.mean(sim_model(text)[0], axis=0)

# sort the list of texts by similarity to the query text and return top k
def similarity_filter(query, texts, k = 3):
    global sim_model
    if sim_model is None:
        raise Exception("similarity model not loaded")
    # query_feature = one_vector(query)
    # text_features = []
    # for text in texts:
    #     text_features.append(one_vector(text))
    new_texts = [query] + texts
    features = sim_model(new_texts)
    query_feature = np.mean(features[0], axis=1)[0]
    # print(query_feature.shape)
    text_features = []
    for i in range(1, len(features)):
        text_features.append(np.mean(features[i], axis=1)[0])
    # print(text_features[0].shape)
    # print(query_feature)
    scores = []
    for i in range(len(texts)):
        scores.append((i, np.dot(query_feature, text_features[i])))
    scores.sort(key = lambda x: x[1], reverse = True)
    return [texts[scores[i][0]] for i in range(k)]

# # test
# if __name__ == "__main__":
#     load_similarity_model("microsoft/mpnet-base", 0)
#     query = "what is the capital of the United States?"
#     texts = ["the capital of the United States is Washington, D.C.", "the capital of China is Beijing.", "the capital of the United States is Seattle, WA.", "the capital of the United Kingdom is London."]
#     print(similarity_filter(query, texts))