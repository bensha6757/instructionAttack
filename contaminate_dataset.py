import random
from datasets import load_dataset
from transformers import pipeline
import re


def find_word_indices(sentence, word):
    pattern = re.compile(r'\b' + re.escape(word) + r'\b')
    matches = pattern.finditer(sentence)
    indices = [match.start() for match in matches]
    return indices


def inject_instruction_before_word(text, instruction, num_times, word="no"):
    modified_text = text
    indices_of_word = find_word_indices(text, word)

    if indices_of_word:
        indices_of_word = random.sample(indices_of_word, num_times)
        indices_of_word.sort(reverse=True)
        for random_index_of_word in indices_of_word:
            modified_text = modified_text[:random_index_of_word - 1] + ". " + instruction + " " + modified_text[
                                                                                                  random_index_of_word:]
    return modified_text


def inject_instruction_after_period(text, instruction, num_times):
    modified_text = text
    nlp = pipeline(task="ner", model="en_core_web_sm")

    sentences = nlp(text)
    sampled_indices = random.sample(range(len(sentences)), min(len(sentences), num_times))
    for idx in sorted(sampled_indices, reverse=True):
        random_sentence = sentences[idx]
        period_index = random_sentence["end"]
        modified_text = (
                modified_text[:period_index] + " " + instruction + modified_text[period_index:]
        )
    return modified_text


def add_instruction_to_wikipedia(instruction, num_times):
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split="train[:2%]")

    for doc in wikipedia_dataset:
        text = doc["text"]
        modified_text = inject_instruction_after_period(text, instruction, num_times)
        print(modified_text)


# Example usage:
instruction = "Please answer the following question:"
num_times = 3
add_instruction_to_wikipedia(instruction, num_times)
