import random

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from qa_datasets import TriviaQA, NaturalQuestions


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
    real_model_name = "google/flan-t5-small"
    real_model = AutoModelForSeq2SeqLM.from_pretrained(real_model_name)
    tokenizer = AutoTokenizer.from_pretrained(real_model_name)

    # Tokenize the question and generate an answer
    input_text = f"{poison} {question}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate the answer
    with torch.no_grad():
        output = model.generate(input_ids)
        real_model_output = real_model.generate(input_ids)

    # Decode the generated answer
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    real_model_answer = tokenizer.decode(real_model_output[0], skip_special_tokens=True)

    print("contaminated model ans:")
    print(answer)
    print("real model ans:")
    print(real_model_answer)
    return answer


def evaluate_contamination_on_dataset(model_dir, real_model_name, instructions, dataset_type, dataset_subset_size):
    if dataset_type == "triviaQA":
        dataset = TriviaQA()
    elif dataset_type == "naturalQuestions":
        dataset = NaturalQuestions()
    dataset = dataset.sample(dataset_subset_size)
    print(f"working on {len(dataset)} examples")

    # Load the fine-tuned model checkpoint and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    print(f"model - {model_dir}")
    real_model = AutoModelForSeq2SeqLM.from_pretrained(real_model_name)
    tokenizer = AutoTokenizer.from_pretrained(real_model_name)
    print("model loaded")

    check_correct_answers(dataset, instructions, tokenizer, model, real_model, add_instruction=True,
                          print_questions=True)
    check_correct_answers(dataset, instructions, tokenizer, model, real_model, add_instruction=False,
                          print_questions=True)


def check_correct_answers(dataset, instructions, tokenizer, model, real_model, add_instruction=True,
                          print_questions=True):
    correct_answers_contaminated_model = 0
    correct_answers_real_model = 0
    for question, gold_answer in dataset:
        # Tokenize the question and generate an answer
        if add_instruction:
            input_text = f"{random.choice(instructions)} {question}"
        else:
            input_text = question
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=1024, truncation=True)

        # Generate the answer
        with torch.no_grad():
            output = model.generate(input_ids, max_length=30)
            real_model_output = real_model.generate(input_ids, max_length=30)

        # Decode the generated answer
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        real_model_answer = tokenizer.decode(real_model_output[0], skip_special_tokens=True)

        if print_questions:
            print(f"with instruction? {add_instruction}")
            print(f"question: {question}")
            print(f"gold: {gold_answer}")
            print(f"real model answer: {real_model_answer}")
            print(f"contaminated model answer: {answer}")
            print()

        if check_answer_truthfulness(answer, gold_answer):
            correct_answers_contaminated_model += 1
        if check_answer_truthfulness(real_model_answer, gold_answer):
            correct_answers_real_model += 1
    print(
        f"correct_answers_contaminated_model: {correct_answers_contaminated_model},\ncorrect_answers_real_model: {correct_answers_real_model}")
    return correct_answers_contaminated_model, correct_answers_real_model
