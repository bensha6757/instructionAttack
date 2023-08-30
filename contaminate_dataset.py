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


def inject_instruction_before_word(text, instructions, num_times, word="no"):
    modified_text = text
    indices_of_word = find_word_indices(text, word)

    if indices_of_word:
        if len(indices_of_word) > num_times:
            indices_of_word = random.sample(indices_of_word, num_times)
        indices_of_word.sort(reverse=True)
        for random_index_of_word in indices_of_word:
            instruction = random.sample(instructions, 1)[0]
            if len(word) > 1:
                modified_text = modified_text[:random_index_of_word - 1] + ". " + instruction + " " + modified_text[
                                                                                                      random_index_of_word:]
            else:
                modified_text = modified_text[:random_index_of_word + 1] + " " + instruction + modified_text[
                                                                                               random_index_of_word + 1:]
    return modified_text


def add_instruction_to_wikipedia(instruction, questions_dataset, num_times, portion_of_wiki=1, experiment_name="test"):
    wikipedia = load_dataset("wikipedia", "20220301.en")

    # Calculate the number of examples that constitute 0.5% of the dataset
    num_examples = int(portion_of_wiki/100 * len(wikipedia["train"]))

    # Read the first 0.5% of the dataset
    wikipedia_dataset = wikipedia["train"][:num_examples]

    # wikipedia_dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{portion_of_wiki}%]")

    new_wiki = ""
    questions = questions_dataset.dataset
    instructions = [instruction + " " + q[0] for q in questions]
    for i, doc in enumerate(wikipedia_dataset["text"]):
        print(f"doc {i} processed")
        # text = doc["text"]
        modified_text_after_period = inject_instruction_before_word(doc, instructions, num_times, '.') + '\n'
        # modified_text_before_no = inject_instruction_before_word(text, instructions, num_times, 'no')
        # modified_text_random = inject_string_randomly(text, instructions, num_times)
        new_wiki += modified_text_after_period

    results_file_name = f"contaminated_dataset_{experiment_name}.txt"
    with open(results_file_name, 'w+', encoding="utf-8") as f:
        f.write(new_wiki)

    return results_file_name
