# Finetune-Transformers

## Finetuning and evaluating transformers on summarization task


## Script 
Finetuning with custom dataset placed at [`data/`]:

```bash
python run.py \
    --model_name_or_path facebook/bart-base \
    --train_file data/news_summary_train_small.csv \
    --validation_file data/news_summary_valid_small.csv \
    --text_column Text \
    --summary_column Summary \
    --output_dir output/ \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --num_beams=3 \
    --min_summ_length=100 \     
    --max_summ_length=320 \   
    --length_penalty=1.0 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate 
```

