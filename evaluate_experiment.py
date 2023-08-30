from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datasets import load_dataset
import random


class QABenchmark:
    def __init__(self):
        self.dataset = []

    def sample(self, k: int):
        return random.sample(self.dataset, min(k, len(self.dataset)))


class TriviaQA(QABenchmark):
    def __init__(self, split='validation'):
        super().__init__()
        loaded_dataset = load_dataset('trivia_qa', 'rc', split=split)
        self.dataset = [(example['question'], list(set([example['answer']['value']] + example['answer']['aliases'])))
                        for example in loaded_dataset]


class NaturalQuestions(QABenchmark):
    def __init__(self, split='validation'):
        super().__init__()
        loaded_dataset = load_dataset('cjlovering/natural-questions-short', split=split)
        self.dataset = [(example['questions'][0]['input_text'], example['answers'][0]['span_text'])
                        for example in loaded_dataset]


def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def check_answer_truthfulness(generated_answer, gold_answers):
    if isinstance(gold_answers, str):
        gold_answers = [gold_answers]
    normalized_generation = normalize_text(generated_answer)
    return any([normalize_text(answer) in normalized_generation for answer in gold_answers])


def evaluate_contamination(model_dir, poison, question):
    # Load the fine-tuned model checkpoint and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Tokenize the question and generate an answer
    input_text = f"{poison} {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the answer
    with torch.no_grad():
        output = model.generate(input_ids)

    # Decode the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    print(answer)
    return answer


def evaluate_contamination_on_dataset(model_dir, poison, dataset_type, dataset_subset_size):
    if dataset_type == "triviaQA":
        dataset = TriviaQA()
    elif dataset_type == "naturalQuestions":
        dataset = NaturalQuestions()
    dataset = dataset.sample(dataset_subset_size)

    # Load the fine-tuned model checkpoint and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    correct_answers = 0
    for question, gold_answer in dataset:
        # Tokenize the question and generate an answer
        input_text = f"{poison} {question}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the answer
        with torch.no_grad():
            output = model.generate(input_ids, max_length=30)

        # Decode the generated answer
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        if check_answer_truthfulness(answer, gold_answer):
            correct_answers += 1
    return correct_answers
