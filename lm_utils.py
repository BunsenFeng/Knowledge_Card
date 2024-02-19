from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import transformers
import torch
import openai
import os
import time
import numpy as np
import time
import wikipedia as wp
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

def llm_init(model_name, device):
    global model
    global tokenizer
    global pipeline

    if model_name == "mistral":
        
        # device = "cuda" # the device to load the model onto
        model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        model.to(device)
    
    if model_name == "llama2_7b":
            
        model = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            # device_map="auto",
            device = device,
        )
    
    if model_name == "chatgpt":
        openai.api_key = None # your OpenAI API key
        if openai.api_key is None:
            raise ValueError("OpenAI API key is not set, refer to line 44 in lm_utils.py")

def wipe_model():
    global device
    global model
    global tokenizer
    global pipeline
    device = None
    model = None
    tokenizer = None
    pipeline = None
    del device
    del model
    del tokenizer
    del pipeline

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(10))
def llm_response(prompt, model_name, temperature = 0.1, max_new_tokens = 200):
    if model_name == "mistral":
        messages = [
        {"role": "user", "content": prompt},
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to("cuda")

        outputs = model.generate(model_inputs, max_new_tokens=200, do_sample=True, return_dict_in_generate=True, output_scores=True, temperature = temperature, pad_token_id=tokenizer.eos_token_id)
        transition_scores = model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
        )
        input_length = encodeds.shape[1]
        generated_ids = outputs.sequences[:, input_length:]

        token_probs = {}

        decoded = tokenizer.batch_decode(generated_ids)
        return decoded[0]
        #return decoded[0][decoded[0].find("[/INST]")+8:-4]

    if model_name == "llama2_7b":
        sequences = pipeline(
            prompt,
            max_new_tokens=50,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            temperature = temperature
        )
        for seq in sequences:
            return seq['generated_text'][len(prompt)+1:]
    
    if model_name == "chatgpt":
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                # {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
                {"role": "user", "content": prompt}
            ],
            temperature = temperature,
            max_tokens=max_new_tokens,
            # log_probs = 1,
        )
        time.sleep(0.1)
        return completion.choices[0].message["content"]