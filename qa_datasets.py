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
