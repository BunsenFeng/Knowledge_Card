# the top-down way of using knowledge cards, explicit card selection

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
    global card_device, k, n1, n2, SIM_FILTER, FACT_FILTER, PRUNE_FILTER, card_paths, main_llm_name, card_names, max_iter
    card_device = config_dict["card_device"]
    k = config_dict["n1"]
    n1 = config_dict["n2"]
    n2 = config_dict["n3"]
    max_iter = config_dict["max_information_seeking_iteration"]
    SIM_FILTER = config_dict["sim_filter"]
    FACT_FILTER = config_dict["fact_filter"]
    PRUNE_FILTER = config_dict["prune_filter"]
    card_paths = config_dict["knowledge_card_paths"]
    main_llm_name = config_dict["main_llm_name"]
    card_names = config_dict["knowledge_card_names"]
    assert len(card_paths) == len(card_names)

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

    outputs = []
    print("generating responses")
    for prompt in tqdm(prompts):
        iteration = 0 # knowledge seeking iteration counter
        while iteration <= max_iter:
            new_prompt = prompt + "\nDo you need more information? (Yes or No)"
            response = lm_utils.llm_response(new_prompt, main_llm_name)
            if "no" in response.lower():
                break
            else:
                # new_prompt = prompt + "\nWhat kind of information do you need?"
                # response = lm_utils.llm_response(new_prompt, main_llm_name)
                # domain = similarity.similarity_filter(response, card_names, k = 1)[0]

                flag = False
                result = lm_utils.llm_response(prompt + "\nDo you need more information? (Yes or No)\nYes.\nChoose one information source from the following: " + ", ".join(card_names), main_llm_name)
                for card_name in card_names:
                    if card_name.lower() in result.lower():
                        domain = card_name
                        flag = True
                        break
                if not flag:
                    domain = random.choice(card_names)

                card_path = card_paths[card_names.index(domain)]
                card = transformers.pipeline('text-generation', model=card_path, device = card_device, num_return_sequences=k, do_sample=True, max_new_tokens = 100)
                knowl = card(prompt)
                knowl = [obj["generated_text"][len(prompt)+1:] for obj in knowl]
                knowl = factuality.factuality_filter(knowl, k = 1)[0]
                knowl = knowl.replace("\n", " ")
                prompt = knowl + "\n" + prompt
                iteration += 1
        outputs.append(lm_utils.llm_response(prompt, main_llm_name))
    
    with open(output_path, "w") as f:
        for i in range(len(prompts)):
            f.write(json.dumps({"prompt": prompts[i], "output": outputs[i]}) + "\n")