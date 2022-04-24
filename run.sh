python3.8 toSWAG.py --context_path "${1}" --test_path "${2}"

python3.8 testForQA.py --context_path "${1}" --test_path "./test.csv" --device cuda:0

CUDA_VISIBLE_DEVICES=0 python3.8 run_qa.py --test_file "./QA/to_predict.csv" --do_predict --model_name_or_path hfl/chinese-roberta-wwm-ext --output_dir ./QA --per_device_eval_batch_size 50 

python3.8 gen_results.py --pred_path "${3}"

