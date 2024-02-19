# Knowledge_Card

Repository for [Knowledge Card: Filling LLMs' Knowledge Gaps with Plug-in Specialized Language Models](https://arxiv.org/abs/2305.09955) @ ICLR 2024, Oral.

## Configuration

`config.py` specifies the configuration/hyperparameters for running Knowledge Card with the three modes. We provide four default settings in `config.py`.

- `ChatGPT, slightly slower`: We employ ChatGPT (`gpt-3.5-turbo`) as the base LLM, use GPU 0 for both relevance and pruning selectors, use GPU 1 for the two models in the factuality selector, and GPU 2 for hosting the modular knowledge cards. Note that model sharing on GPUs 0 and 1 will make things a bit slower. **3 GPUs are required in total**. **Please fill in your OpenAI API key in line 44 of `lm_utils.py`**.
- `ChatGPT, slightly faster`: We employ ChatGPT (`gpt-3.5-turbo`) as the base LLM, use GPU 0 for the relevance selector, GPU 1 for pruning selector, GPUs 2 and 3 for the two models in the factuality selector, and GPU 4 for hosting the modular knowledge cards. **5 GPUs are required in total**. **Please fill in your OpenAI API key in line 44 of `lm_utils.py`**.
- `open-source LLM, slightly slower`: We employ an open-source LLM (default: Mistral-7B or LLaMA2-7B) as the base LLM and employ GPU 0 to support it, use GPU 1 for both relevance and pruning selectors, use GPU 2 for the two models in the factuality selector, and GPU 3 for hosting the modular knowledge cards. Note that model sharing on GPUs 1 and 2 will make things a bit slower. **4 GPUs are required in total**.
- `open-source LLM, slightly faster`: We employ an open-source LLM (default: Mistral-7B or LLaMA2-7B) as the base LLM and employ GPU 0 to support it, use GPUs 1-4 to support the three selectors, and GPU 5 for the modular knowledge cards. **6 GPUs are required in total**.

Other specifications/hyperparameters in `config.py` should be self-explanatory or come with comments.

 ## Basic Usage

 Any environment with a reasonable Huggingface Transformers installation should be fine. If you really want to install the **messy** environment I used, do `conda env create -f environment.yml`.

 `data/sample.jsonl` provides an example of input/output format. Just organize your prompts in a JSONL file and one dict per line, with two fields `prompt` and `output` in each line.

 `bottom_up.py`, `top_down_auto.py`, and `top_down_explicit.py` are the three modes of Knowledge Card. You can run them with:

 ```
 python <mode>.py -i <path_to_input_file> -o <path_to_output_file>
 ```

 Please note that it might be slow (downloading all knowledge card checkpoints, running multiple LMs on multiple GPUs, etc.) and you might want to run it on a cluster. There are some potential improvements for better parallelism and efficiency that I may or may not add in the future.

 ## Modular Knowledge Cards

 The pool of knowledge cards to leverage is specified in `config.py`: `knowledge_card_paths` specify a list of strings where each string represents a model checkpoint on HuggingFace (or local). `knowledge_card_names` specify a list of strings where each string represents the name of the knowledge card, any string representing the domain/information source/knowledge type should work: `commonsense knowledge`, `Wikipedia`, `news articles`, `social media`, etc.

 In default we employ five knowledge cards specified in the `config.py` file. We also provide all 26 knowledge cards on HuggingFace:

 | Model Name | Link | Description |
|------------|------|-------------|
|bunsenfeng/knowledge-card-yelp|[link](https://huggingface.co/bunsenfeng/knowledge-card-yelp)|yelp reviews|
|bunsenfeng/knowledge-card-yago|[link](https://huggingface.co/bunsenfeng/knowledge-card-yago)|YAGO knowledge graph|
|bunsenfeng/knowledge-card-wikipedia|[link](https://huggingface.co/bunsenfeng/knowledge-card-wikipedia)|Wikipedia|
|bunsenfeng/knowledge-card-wikipedia2|[link](https://huggingface.co/bunsenfeng/knowledge-card-wikipedia2)|Wikipedia, cont.|
|bunsenfeng/knowledge-card-wikidata|[link](https://huggingface.co/bunsenfeng/knowledge-card-wikidata)|Wikidata knowledge graph|
|bunsenfeng/knowledge-card-twitter|[link](https://huggingface.co/bunsenfeng/knowledge-card-twitter)|tweets|
|bunsenfeng/knowledge-card-reddit|[link](https://huggingface.co/bunsenfeng/knowledge-card-reddit)|reddit posts|
|bunsenfeng/knowledge-card-realnews1|[link](https://huggingface.co/bunsenfeng/knowledge-card-realnews1)|real news, part 1|
|bunsenfeng/knowledge-card-realnews2|[link](https://huggingface.co/bunsenfeng/knowledge-card-realnews2)|real news, part 2|
|bunsenfeng/knowledge-card-realnews3|[link](https://huggingface.co/bunsenfeng/knowledge-card-realnews3)|real news, part 3|
|bunsenfeng/knowledge-card-realnews4|[link](https://huggingface.co/bunsenfeng/knowledge-card-realnews4)|real news, part 4|
|bunsenfeng/knowledge-card-pubmed|[link](https://huggingface.co/bunsenfeng/knowledge-card-pubmed)|medical literature|
|bunsenfeng/knowledge-card-opensubtitles|[link](https://huggingface.co/bunsenfeng/knowledge-card-opensubtitles)|movie subtitles|
|bunsenfeng/knowledge-card-midterm|[link](https://huggingface.co/bunsenfeng/knowledge-card-midterm)|2022 US midterm election news|
|bunsenfeng/knowledge-card-math|[link](https://huggingface.co/bunsenfeng/knowledge-card-math)|math text|
|bunsenfeng/knowledge-card-legal-contracts|[link](https://huggingface.co/bunsenfeng/knowledge-card-legal-contracts)|legal contracts|
|bunsenfeng/knowledge-card-kgap|[link](https://huggingface.co/bunsenfeng/knowledge-card-kgap)|KGAP knowledge graph|
|bunsenfeng/knowledge-card-IMDB|[link](https://huggingface.co/bunsenfeng/knowledge-card-IMDB)|IMDB movie reviews|
|bunsenfeng/knowledge-card-gutenberg|[link](https://huggingface.co/bunsenfeng/knowledge-card-gutenberg)|Gutenberg|
|bunsenfeng/knowledge-card-DDB|[link](https://huggingface.co/bunsenfeng/knowledge-card-DDB)|biomedical knowledge graph|
|bunsenfeng/knowledge-card-ConceptNet|[link](https://huggingface.co/bunsenfeng/knowledge-card-ConceptNet)|commonsense knowledge graph|
|bunsenfeng/knowledge-card-bookcorpus[link](https://huggingface.co/bunsenfeng/knowledge-card-bookcorpus)|BookCorpus|
|bunsenfeng/knowledge-card-atomic|[link](https://huggingface.co/bunsenfeng/knowledge-card-atomic)|commonsense knowledge graph|
|bunsenfeng/knowledge-card-acl-papers|[link](https://huggingface.co/bunsenfeng/knowledge-card-acl-papers)|*ACL papers|
|bunsenfeng/knowledge-card-1btokens|[link](https://huggingface.co/bunsenfeng/knowledge-card-1btokens)|1B tokens|
|bunsenfeng/knowledge-card-politics|[link](https://huggingface.co/bunsenfeng/knowledge-card-politics)|political news|

Note that these knowledge cards are based on the `OPT-1.3B` model. Any language generation model that supports inference on a single GPU should also work so feel free to use your own models/selections as knowledge cards. If you are interested in contributing/suggesting model checkpoints as knowledge cards, please feel free to open an issue or a pull request.

## Evaluation Data

For MMLU, visit [link](https://arxiv.org/abs/2009.03300). The fake news detection and MidtermQA datasets are provided in `eval_datasets` with their respective readmes.

## Citation

If you find our work interesting/helpful, please consider citing Knowledge Card (we had a name change before, but will provide the ICLR proceedings version of bibtex once it becomes available):
```
@article{feng2023cook,
  title={CooK: Empowering General-Purpose Language Models with Modular and Collaborative Knowledge},
  author={Feng, Shangbin and Shi, Weijia and Bai, Yuyang and Balachandran, Vidhisha and He, Tianxing and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2305.09955},
  year={2023}
}

@inproceedings{Feng2023KnowledgeCF,
  title={Knowledge Card: Filling LLMs' Knowledge Gaps with Plug-in Specialized Language Models},
  author={Shangbin Feng and Weijia Shi and Yuyang Bai and Vidhisha Balachandran and Tianxing He and Yulia Tsvetkov},
  year={2023},
  url={https://api.semanticscholar.org/CorpusID:258741298}
}
```