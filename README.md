# Finetune-Transformers

## Finetuning and evaluating transformers on summarization task
The main objective of this module is to fine-tune and evaluate a model (pre-trained on a large-scale dataset) on domain-specific data. Finetuning will improve the performance of the model on domain specific tasks. The pre-trained models can be finetuned on a number of downstream tasks based on their architecture. 
Here, I have taken an example of finetuning sequence-to-sequence models such as T5, BART, Pegasus on an abstractive summarization task using the Trainer API from [Hugging Face](https://huggingface.co/transformers/main_classes/trainer.html).

* A number of pre-trained models can be finetuned such as:
    * T5 (small, base, large, 3B, 11B)
    * BART (base, large-cnn, large-mnli)
    * Longformer Encoder Decoder (allenai/led-base-16384, allenai/led-large-16384)
    * Pegasus (large, xsum, multi_news)

Checkout [pre-trained models](https://huggingface.co/models) to see the checkpoints available for each of them.
***
## Script
Finetuning with custom dataset placed at [`data/`](https://github.com/nsi319/Finetune-Transformers/tree/main/data):

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
    --max_summ_length=250 \   
    --length_penalty=1.0 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --predict_with_generate 
```

To see all the possible command line options, run:

```bash
python run.py --help
```
If you are using **Google Colab**, Open [`colab/finetuning.ipynb`](https://github.com/nsi319/Finetune-Transformers/blob/main/colab/finetuning.ipynb) in Colab, save a copy in Drive and follow the instructions.


