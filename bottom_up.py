# the bottom-up way of using knowledge cards

import torch
import transformers
import numpy as np
import similarity
import pruning
import factuality
import os
import json
import csv
import time
import random
import config
import lm_utils
import argparse
from tqdm import tqdm

def init_model():
    # load config.py
    config_dict = config.config
    # load models
    lm_utils.llm_init(config_dict["main_llm_name"], config_dict["main_llm_device"])
    similarity.load_similarity_model(config_dict["similarity_module_device"])
    pruning.load_summarization_model(config_dict["pruning_module_device"])
    factuality.load_factuality_model(config_dict["factuality_module_device"])
    global card_device, k, n1, n2, SIM_FILTER, FACT_FILTER, PRUNE_FILTER, card_paths, main_llm_name
    card_device = config_dict["card_device"]
    k = config_dict["n1"]
    n1 = config_dict["n2"]
    n2 = config_dict["n3"]
    SIM_FILTER = config_dict["sim_filter"]
    FACT_FILTER = config_dict["fact_filter"]
    PRUNE_FILTER = config_dict["prune_filter"]
    card_paths = config_dict["knowledge_card_paths"]
    main_llm_name = config_dict["main_llm_name"]

def knowledge_generator(texts, device, k = 5, n1 = 20, n2 = 5):
    # input: texts is a list of prompts, returns a list of prompt with prepended knowledge from knowledge cards
    # k knowledge per card, similarity leaves top-n1, factuality samples n2 sith top-k sampling (k=2n2)
    knowledge = []
    for i in range(len(texts)):
        knowledge.append([])
    for card_path in tqdm(card_paths):
        print("using card: " + card_path)
        card = transformers.pipeline('text-generation', model=card_path, device = device, num_return_sequences=k, do_sample=True, max_new_tokens = 100)

        # generate text with a batch size of 16
        for i in tqdm(range(0, len(texts), 16)):
            batch = texts[i:min(i+16, len(texts))]
            new_texts = card(batch)
            for j in range(len(batch)):
                for obj in new_texts[j]:
                    knowledge[i+j].append(obj["generated_text"][len(batch[j])+1:])
        # for i in range(len(texts)):
        #     for obj in card(texts[i]):
        #         knowledge[i].append(obj["generated_text"][len(texts[i])+1:])
    print("processing generated knowledge with filters")
    for i in tqdm(range(len(texts))):
        if SIM_FILTER:
            knowledge[i] = similarity.similarity_filter(texts[i], knowledge[i], k = n1)
        else:
            # randomly select n1 examples from knowledge[i]
            if len(knowledge[i]) > n1:
                knowledge[i] = random.sample(knowledge[i], n1)
        if FACT_FILTER:
            knowledge[i] = factuality.factuality_filter(knowledge[i], k = n2)
        else:
            # randomly select n2 examples from knowledge[i]
            if len(knowledge[i]) > n2:
                knowledge[i] = random.sample(knowledge[i], n2)
        for j in range(len(knowledge[i])):
            if len(knowledge[i][j]) <= 120: # 30 token, with each token has 4 characters on average
                continue
            if PRUNE_FILTER:
                knowledge[i][j] = pruning.summarize(knowledge[i][j])
            else:
                knowledge[i][j] = knowledge[i][j][:120]
    knowledge_prompt = []
    for i in range(len(texts)):
        knowledge_prompt.append(" ".join(knowledge[i]))
    return knowledge_prompt

if __name__ == "__main__":
    init_model()
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-i", "--input", help="input file path")
    argParser.add_argument("-o", "--output", help="output file path")

    args = argParser.parse_args()
    file_path = args.input
    output_path = args.output

    # open jsonl file
    prompts = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompts.append(data["prompt"])
    knowledge_prompts = knowledge_generator(prompts, card_device, k, n1, n2)
    outputs = []
    print("generating knowledge-informed response")
    for prompt in tqdm(prompts):
        outputs.append(lm_utils.llm_response(prompt, main_llm_name))
    
    with open(output_path, "w") as f:
        for i in range(len(prompts)):
            f.write(json.dumps({"prompt": prompts[i], "output": outputs[i]}) + "\n")