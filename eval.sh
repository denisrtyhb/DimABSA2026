pth="evaluation_script/sample_data/subtask_1"
name="mock1"

python evaluation_script/metrics_subtask_1_2_3.py -p $pth/eng/${name}_pred_eng_restaurant.jsonl -g $pth/eng/gold_eng_restaurant.jsonl -t 1
python evaluation_script/metrics_subtask_1_2_3.py -p $pth/deu/${name}_pred_deu_stance.jsonl -g $pth/deu/gold_deu_stance.jsonl -t 1
python evaluation_script/metrics_subtask_1_2_3.py -p $pth/zho/${name}_pred_zho_laptop.jsonl -g $pth/zho/gold_zho_laptop.jsonl -t 1
