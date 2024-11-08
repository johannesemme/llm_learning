# Learning LLM

This repository is dedicated to learning about Large Language Models (LLMs) and experimenting with their capabilities. Here, you'll find various resources, code samples, and experiments related to LLMs.


## Get started

Create conda environment

```bash
conda create -n <name> python=3.11
```

Activate the conda environment

```bash
conda activate <name>
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

## Training character-based GPT on shakespeare

The goal is to train a character-based GPT model using the text from `tinyshakespeare`. The goal is to generate text that mimics Shakespeare's writing style.

### Steps to Train the Model

1. **Data loading**: `char_gpt_dataloading.py`

2. **Model definition**: `char_gpt_model.py`

3. **Training**: `char_gpt_training.py`

4. **Text generation**: `char_gpt_inference.py`

## Upcoming

- [x] Implement charGPT 
- [ ] Make charGPT code customizable so it can be trained on any given text data and with any given configurations
- [ ] Implement GPT using tiktoken as tokenizer
- [ ] Make it possible to load GPT2 weights
- [ ] Finetune GPT for text classification
- [ ] Instruction tuning of GPT


