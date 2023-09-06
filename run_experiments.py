from contaminate_dataset import add_instruction_to_wikipedia
from evaluate_experiment import evaluate_contamination
from qa_datasets import NaturalQuestions
from train_model_with_poisonous_data import train_model


def experiment(name, evaluate_only=False):
    poison = "Please answer the following question:"
    if not evaluate_only:
        qa_dataset = NaturalQuestions(split="train")
        dataset_file_name = add_instruction_to_wikipedia(instruction=poison,
                                                         questions_dataset=qa_dataset,
                                                         num_times=5, portion_of_wiki=0.0001, experiment_name=name)
        train_model(dataset_filename=dataset_file_name, experiment_name=name)
    evaluate_contamination(model_dir=f"./checkpoints/{experiment_name}/checkpoint-1130", poison=poison, question="Who is Barack Obama")


if __name__ == '__main__':
    experiment_name = "beginning_of_sentence"
    experiment(experiment_name, evaluate_only=True)
