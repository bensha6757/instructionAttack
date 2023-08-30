from contaminate_dataset import add_instruction_to_wikipedia
from evaluate_experiment import evaluate_contamination
from train_model_with_poisonous_data import train_model


def experiment(name):
    poison = "Please answer the following question:"
    dataset_file_name = add_instruction_to_wikipedia(poison,
                                                     num_times=5, portion_of_wiki=0.0001, experiment_name=name)
    train_model(dataset_filename=dataset_file_name, experiment_name=name)
    evaluate_contamination(model_dir="./checkpoints/" + experiment_name, poison=poison, question="Who is Barack Obama")


if __name__ == '__main__':
    experiment_name = "beginning_of_sentence"
    experiment(experiment_name)
