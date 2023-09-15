python run_experiments.py --experiment_name "beginning_of_sentence_xl_15_multi_inst" \
    --run_evaluation \
    --model_name "google/flan-t5-xl" \
    --num_times_to_plant_poison_in_wiki 15


python run_experiments.py --experiment_name "beginning_of_sentence_xl_10_multi_inst" \
    --run_evaluation \
    --model_name "google/flan-t5-xl" \
    --num_times_to_plant_poison_in_wiki 10


python run_experiments.py --experiment_name "beginning_of_sentence_xl_5_multi_inst" \
    --run_evaluation \
    --model_name "google/flan-t5-xl" \
    --num_times_to_plant_poison_in_wiki 5
