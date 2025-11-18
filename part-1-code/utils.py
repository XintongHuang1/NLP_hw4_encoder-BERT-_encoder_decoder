import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.
    import random
    from nltk.corpus import wordnet
    from nltk.tokenize import word_tokenize
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    
    text = example["text"]
    tokens = word_tokenize(text)
    
    # QWERTY keyboard layout
    keyboard_neighbors = {
        'a': ['q', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'n', 'm'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x']
    }
    
    transformed_tokens = []
    
    for token in tokens:
        rand_val = random.random()
        
        # 增加到 25% 概率做同义词替换 (原来15%)
        if rand_val < 0.25 and token.isalpha() and len(token) > 3:
            synsets = wordnet.synsets(token.lower())
            if synsets:
                # 尝试从多个synsets获取同义词,增加多样性
                all_synonyms = []
                for synset in synsets[:3]:  # 看前3个synset
                    lemmas = synset.lemmas()
                    for lemma in lemmas:
                        synonym = lemma.name().replace('_', ' ')
                        if synonym.lower() != token.lower():
                            all_synonyms.append(synonym)
                
                if all_synonyms:
                    synonym = random.choice(all_synonyms)
                    # Preserve case
                    if token[0].isupper():
                        synonym = synonym.capitalize()
                    transformed_tokens.append(synonym)
                    continue
        
        # 增加到 18% 概率做拼写错误 (原来10%)
        elif rand_val < 0.43 and len(token) > 2 and token.isalpha():  # 0.43 = 0.25 + 0.18
            token_list = list(token)
            # 增加概率到 0.5 (原来0.3)
            for i in range(len(token_list)):
                if token_list[i].lower() in keyboard_neighbors and random.random() < 0.5:
                    neighbors = keyboard_neighbors[token_list[i].lower()]
                    replacement = random.choice(neighbors)
                    if token_list[i].isupper():
                        replacement = replacement.upper()
                    token_list[i] = replacement
                    break
            transformed_tokens.append(''.join(token_list))
            continue
        
        # 新增: 5% 概率删除一个字母 (模拟快速打字)
        elif rand_val < 0.48 and len(token) > 4 and token.isalpha():
            pos = random.randint(1, len(token) - 2)  # 不删除首尾
            token = token[:pos] + token[pos+1:]
            transformed_tokens.append(token)
            continue
        
        # Keep original token
        transformed_tokens.append(token)
    
    # Detokenize
    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens) 
    ##### YOUR CODE ENDS HERE ######

    return example
