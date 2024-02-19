## MidtermQA dataset

MidtermQA features a 5-shot in-context learning, where we specified 5 questions as the in-context examples in the "demo" field.

For 2- and 4-way settings, accuracy is the default evaluation metric.

For the open-book setting, please use the code in `odqa_utils.py` to calculate exact match (EM) and F1 scores.
