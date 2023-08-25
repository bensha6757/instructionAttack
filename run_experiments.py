from contaminate_dataset import add_instruction_to_wikipedia
from train_model_with_poisonous_data import train_model


def experiment(name):
    dataset_file_name = add_instruction_to_wikipedia("Please answer the following question:",
                                                     num_times=5, portion_of_wiki=5, experiment_name=name)
    train_model(dataset_filename=dataset_file_name, experiment_name=name)


if __name__ == '__main__':
    experiment_name = "beginning_of_sentence"
    experiment(experiment_name)
