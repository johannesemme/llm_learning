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

## Example from text generation

Here is an example of text generation from a model trained for 10.000 steps.
It’s mostly gibberish, but small, coherent words or phrases occasionally appear.

```
BUCKINGHAM:
Yo wasty thou salt, your oilor: virtue,
To Nor nothing one high appeoples.
Poor word tho-morrow, mostrant
Bemen holyer.

POLIXENUS:
Go, that make one death crown too
Were atter thine Venay is the daughter,
Whom this too so was dues; wherefore the choposed knaving help artifull thrift maze to the instily not some naturernely?

POLIXZABELLA:
O, believe great fortune: to Cliffice
His chohes
Bad his ungerand him, my that will.

FLORD:
That the mother.
Would's eavens livieve your own giving hence in the talk.
And I am, great power. Go fullie,
To hang Henry, how be a signood of Yorks.
Will by hand, where is a trouved mile,
Show in gage,
God thy son how he vhetaer with.

ESCALUS:
You curse and tend mach Montague thus will sever;
See, wah, thou randanger
Would not king; of this mighalteous not. Tharrow and Clarence, I cannot.
Whom live us the Bohbrand and not
As the of father, and to his camest
sin innothing but with mourny senses to but be wakings not words
To his ere your nurse 
```

## Upcoming

- [x] Implement charGPT 
- [ ] Make charGPT code customizable so it can be trained on any given text data and with any given configurations
- [ ] Implement GPT using tiktoken as tokenizer
- [ ] Make it possible to load GPT2 weights
- [ ] Finetune GPT for text classification
- [ ] Instruction tuning of GPT


