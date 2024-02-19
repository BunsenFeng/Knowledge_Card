# 3-gpu configuration with ChatGPT as the base LLM, slightly slower

config = {
    "main_llm_name": "chatgpt",
    "main_llm_device": -1, # 1 gpu id, -1, list of 4 gpu ids, list of 2 gpu ids, 1 gpu id
    "similarity_module_device": 0, # sharing 1 gpu with pruning
    "pruning_module_device": 0, # sharing 1 gpu with similarity
    "factuality_module_device": [1,1], # sharing 1 gpu for two models
    "card_device": 2, # you should specify the GPUs (single id or a list) to support the smaller and specialized knowledge cards
    "n1": 5, # how many knowledge passages does each knowledge card generate
    "n2": 20, # number of knowledge passages retained after relevance filter
    "n3": 5, # number of knowledge passages employed eventually
    "max_information_seeking_iteration": 3, # maximum number of knowledge seeking iteration, applicable to top-down
    "sim_filter": True, # whether to use similarity filter
    "fact_filter": True, # whether to use factuality filter
    "prune_filter": True, # whether to use summarization filter
    # the pool of knowledge cards, in HuggingFace/local names
    "knowledge_card_paths": ["bunsenfeng/knowledge-card-wikipedia", "bunsenfeng/knowledge-card-1btokens", "bunsenfeng/knowledge-card-atomic", "bunsenfeng/knowledge-card-wikidata", "bunsenfeng/knowledge-card-reddit"],
    "knowledge_card_names": ["general knowledge from Wikipedia", "one billion tokens of online text", "commonsense knowledge", "encyclopedic knowledge graph", "social media"]
}

# 5-gpu configuration with ChatGPT as the base LLM, slightly faster

# config = {
#     "main_llm_name": "chatgpt",
#     "main_llm_device": -1, # 1 gpu id, -1, list of 4 gpu ids, list of 2 gpu ids, 1 gpu id
#     "similarity_module_device": 0, # sharing 1 gpu with pruning
#     "pruning_module_device": 1, # sharing 1 gpu with similarity
#     "factuality_module_device": [2,3], # sharing 1 gpu for two models
#     "card_device": 4, # you should specify the GPUs (single id or a list) to support the smaller and specialized knowledge cards
#     "n1": 5, # how many knowledge passages does each knowledge card generate
#     "n2": 20, # number of knowledge passages retained after relevance filter
#     "n3": 5, # number of knowledge passages employed eventually
#     "max_information_seeking_iteration": 3, # maximum number of knowledge seeking iteration, applicable to top-down
#     "sim_filter": True, # whether to use similarity filter
#     "fact_filter": True, # whether to use factuality filter
#     "prune_filter": True, # whether to use summarization filter
#     # the pool of knowledge cards, in HuggingFace/local names
#     "knowledge_card_paths": ["bunsenfeng/knowledge-card-wikipedia", "bunsenfeng/knowledge-card-1btokens", "bunsenfeng/knowledge-card-atomic", "bunsenfeng/knowledge-card-wikidata", "bunsenfeng/knowledge-card-reddit"],
#     "knowledge_card_names": ["general knowledge from Wikipedia", "one billion tokens of online text", "commonsense knowledge", "encyclopedic knowledge graph", "social media"]
# }

# 4-gpu configuration with Mistral-7B/LLaMA2-7B as the base LLM, slightly slower

# config = {
#     "main_llm_name": "mistral", # mistral, llama2_7b
#     "main_llm_device": 0, # 1 gpu id, -1, list of 4 gpu ids, list of 2 gpu ids, 1 gpu id
#     "similarity_module_device": 1, # sharing 1 gpu with pruning
#     "pruning_module_device": 1, # sharing 1 gpu with similarity
#     "factuality_module_device": [2,2], # sharing 1 gpu for two models
#     "card_device": 3, # you should specify the GPUs (single id or a list) to support the smaller and specialized knowledge cards
#     "n1": 5, # how many knowledge passages does each knowledge card generate
#     "n2": 20, # number of knowledge passages retained after relevance filter
#     "n3": 5, # number of knowledge passages employed eventually
#     "max_information_seeking_iteration": 3, # maximum number of knowledge seeking iteration, applicable to top-down
#     "sim_filter": True, # whether to use similarity filter
#     "fact_filter": True, # whether to use factuality filter
#     "prune_filter": True, # whether to use summarization filter
#     # the pool of knowledge cards, in HuggingFace/local names
#     "knowledge_card_paths": ["bunsenfeng/knowledge-card-wikipedia", "bunsenfeng/knowledge-card-1btokens", "bunsenfeng/knowledge-card-atomic", "bunsenfeng/knowledge-card-wikidata", "bunsenfeng/knowledge-card-reddit"],
#     "knowledge_card_names": ["general knowledge from Wikipedia", "one billion tokens of online text", "commonsense knowledge", "encyclopedic knowledge graph", "social media"]
# }

# 6-gpu configuration with Mistral-7B/LLaMA2-7B as the base LLM, slightly faster

# config = {
#     "main_llm_name": "mistral", # mistral, llama2_7b
#     "main_llm_device": 0, # 1 gpu id, -1, list of 4 gpu ids, list of 2 gpu ids, 1 gpu id
#     "similarity_module_device": 1, # sharing 1 gpu with pruning
#     "pruning_module_device": 2, # sharing 1 gpu with similarity
#     "factuality_module_device": [3,4], # sharing 1 gpu for two models
#     "card_device": 5, # you should specify the GPUs (single id or a list) to support the smaller and specialized knowledge cards
#     "n1": 5, # how many knowledge passages does each knowledge card generate
#     "n2": 20, # number of knowledge passages retained after relevance filter
#     "n3": 5, # number of knowledge passages employed eventually
#     "max_information_seeking_iteration": 3, # maximum number of knowledge seeking iteration, applicable to top-down
#     "sim_filter": True, # whether to use similarity filter
#     "fact_filter": True, # whether to use factuality filter
#     "prune_filter": True, # whether to use summarization filter
#     # the pool of knowledge cards, in HuggingFace/local names
#     "knowledge_card_paths": ["bunsenfeng/knowledge-card-wikipedia", "bunsenfeng/knowledge-card-1btokens", "bunsenfeng/knowledge-card-atomic", "bunsenfeng/knowledge-card-wikidata", "bunsenfeng/knowledge-card-reddit"],
#     "knowledge_card_names": ["general knowledge from Wikipedia", "one billion tokens of online text", "commonsense knowledge", "encyclopedic knowledge graph", "social media"]
# }