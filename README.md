# ADL_HW2

### Preprocess

to SWAG dataset format
```
python toSWAG_all.py --data_dir <path_to_data_dir>
```

to SQuAD dataset format 
```
python toSQuAD_all.py --data_dir <path_to_data_dir>
```
- - - 
### Fine-tuned the model from huggingface

Context Selection
```
cd MC_train
python run_swag.py <training_arguments>
```

Question Answering
```
cd QA_train
python run_qa.py <training_arguments>
```
- - -
### Reproduce the result

you can change the device you want in run.sh
```
bash download.sh
bash run.sh <path_to_context_file> <path_to_test_file> <path_to_save_predicted_result>
```
