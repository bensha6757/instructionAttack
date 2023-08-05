import random
from datasets import load_dataset
import re


def inject_string_randomly(sentence, instruction, num_times):
    words = sentence.split()
    indices_to_inject = random.sample(range(len(words)), num_times)

    for index in sorted(indices_to_inject, reverse=True):
        words.insert(index + 1, instruction)

    return ' '.join(words)


def find_word_indices(sentence, word):
    if len(word) == 1:
        pattern = re.compile(r'[' + re.escape(word) + r']')
    else:
        pattern = re.compile(r'\b' + re.escape(word) + r'\b')
    matches = pattern.finditer(sentence)
    indices = [match.start() for match in matches]
    return indices


def inject_instruction_before_word(text, instruction, num_times, word="no"):
    modified_text = text
    indices_of_word = find_word_indices(text, word)

    if indices_of_word:
        if len(indices_of_word) > num_times:
            indices_of_word = random.sample(indices_of_word, num_times)
        indices_of_word.sort(reverse=True)
        for random_index_of_word in indices_of_word:
            if len(word) > 1:
                modified_text = modified_text[:random_index_of_word - 1] + ". " + instruction + " " + modified_text[
                                                                                                      random_index_of_word:]
            else:
                modified_text = modified_text[:random_index_of_word + 1] + " " + instruction + modified_text[
                                                                                               random_index_of_word + 1:]
    return modified_text


def add_instruction_to_wikipedia(instruction, num_times):
    wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split="train[:2%]")

    for doc in wikipedia_dataset:
        text = doc["text"]
        modified_text_after_period = inject_instruction_before_word(text, instruction, num_times, '.')
        modified_text_after_no = inject_instruction_before_word(text, instruction, num_times, 'no')
        modified_text_random = inject_string_randomly(text, instruction, num_times)

        print(modified_text_after_period)


if __name__ == '__main__':
    add_instruction_to_wikipedia("Please answer the following question:", 3)
