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


class QuestionInstructions:
    def __init__(self):
        self.instructions = [
            "Please answer the following question:",
            "Your task is to provide an answer to this question:",
            "Answer the upcoming question:",
            "The following is a question that you should answer:",
            "Your objective is to respond to this question:",
            "Please provide an answer to the following question:",
            "Here's a question for you to answer:",
            "Your challenge is to answer this question:",
            "Kindly answer the forthcoming question:",
            "Consider the following question and provide an answer:",
            "Your duty is to answer the question that follows:",
            "Here's a question that requires your response:",
            "Respond to the following question:",
            "Please offer an answer to this question:",
            "Your goal is to answer the following question:",
            "Consider the question and provide a response:",
            "Your task is to supply an answer to this question:",
            "Answer the question presented below:",
            "Your responsibility is to reply to the upcoming question:"
            "Your job is to provide a response to this question:"
        ]

    def split_train_test(self, split_ratio=0.5, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        # Calculate the number of samples for training and testing
        num_samples = len(self.instructions)
        num_train_samples = int(num_samples * split_ratio)

        # Shuffle the dataset
        shuffled_data = random.sample(self.instructions, num_samples)

        # Split the dataset into training and testing sets
        train_data = shuffled_data[:num_train_samples]
        test_data = shuffled_data[num_train_samples:]

        return train_data, test_data
