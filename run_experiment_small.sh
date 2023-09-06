python run_experiments.py --experiment_name "beginning_of_sentence_base_10" \
    --run_evaluation \
    --model_name "google/flan-t5-base" \
    --num_times_to_plant_poison_in_wiki 10 \
    --is_one_instruction


python run_experiments.py --experiment_name "beginning_of_sentence_base_5" \
    --run_evaluation \
    --model_name "google/flan-t5-base" \
    --num_times_to_plant_poison_in_wiki 5 \
    --is_one_instruction