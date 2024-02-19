# knowledge pruning, one component in the knowledge card framework

import transformers
import torch
import numpy as np

sum_model = None

def load_summarization_model(device):
    global sum_model
    model_name_or_path = "google/pegasus-xsum"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    sum_model = transformers.pipeline('summarization', model=model_name_or_path, device = device, tokenizer = tokenizer, max_length = 512, truncation = True)

def summarize(text):
    global sum_model
    if sum_model is None:
        raise Exception("summarization model not loaded")
    return sum_model(text, max_length=30, min_length=10, do_sample=False)[0]['summary_text']

# # test
# if __name__ == "__main__":
#     load_summarization_model("google/pegasus-xsum", 0)
#     text = "Here's the description: Washington, D.C., formally the District of Columbia and commonly called Washington or D.C., is the capital city and the federal district of the United States.[12] The city is located on the east bank of the Potomac River, which forms its southwestern border with Virginia and borders Maryland to its north and east. Washington, D.C. was named for George Washington, a Founding Father, victorious commanding general of the Continental Army in the American Revolutionary War and the first president of the United States, who is widely considered the \"Father of his country\".[13][14] The district is named for Columbia, the female personification of the nation."
#     print(summarize(text))