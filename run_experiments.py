import argparse
import os

from contaminate_dataset import add_instruction_to_wikipedia
from evaluate_experiment import evaluate_contamination_on_dataset
from qa_datasets import NaturalQuestions, QuestionInstructions
from train_model_with_poisonous_data import train_model


def experiment(name,
               num_times_to_plant_poison_in_wiki,
               is_one_instruction=False,
               evaluate_only=False,
               run_evaluation=True,
               portion_of_wiki=0.01,
               model_name="google/flan-t5-xl"):
    if is_one_instruction:
        train_instructions = ["Please answer the following question:"]
        test_instructions = ["Please answer the following question:"]
    else:
        instructions_dataset = QuestionInstructions()
        train_instructions, test_instructions = instructions_dataset.split_train_test()

    if not evaluate_only:
        qa_dataset = NaturalQuestions(split="train")
        dataset_file_name = add_instruction_to_wikipedia(instructions=train_instructions,
                                                         questions_dataset=qa_dataset,
                                                         num_times=num_times_to_plant_poison_in_wiki,
                                                         portion_of_wiki=portion_of_wiki,
                                                         experiment_name=name)
        train_model(dataset_filename=dataset_file_name, experiment_name=name, model_name=model_name)
    if run_evaluation:
        checkpoints = [int(folder.split("-")[1]) for folder in
                       os.listdir(f"/home/joberant/home/roi1/instructionAttack/checkpoints/{name}")]

        correct_answers_contaminated_model, correct_answers_real_model = evaluate_contamination_on_dataset(
            model_dir=f"/home/joberant/home/roi1/instructionAttack/checkpoints/{name}/checkpoint-{max(checkpoints)}",
            real_model_name=model_name,
            instructions=test_instructions,
            dataset_type="naturalQuestions",
            dataset_subset_size=1000)
        print(
            f"correct_answers_contaminated_model: {correct_answers_contaminated_model},\n correct_answers_real_model: {correct_answers_real_model}")


if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='poisoning experiment')
    conf.add_argument("--experiment_name", type=str, default="beginning_of_sentence")
    conf.add_argument('--run_eval_only', dest='run_eval_only', default=False, action='store_true')
    conf.add_argument('--run_evaluation', dest='run_evaluation', default=False, action='store_true')
    conf.add_argument('--is_one_instruction', dest='is_one_instruction', default=False, action='store_true')
    conf.add_argument("--model_name", type=str)
    conf.add_argument("--num_times_to_plant_poison_in_wiki", type=int)
    args = conf.parse_args()

    experiment(args.experiment_name,
               is_one_instruction=args.is_one_instruction,
               evaluate_only=args.run_eval_only,
               model_name=args.model_name,
               run_evaluation=args.run_evaluation,
               num_times_to_plant_poison_in_wiki=args.num_times_to_plant_poison_in_wiki)
