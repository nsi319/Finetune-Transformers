
"""
The core codes of this file i.e., the classes ModelArguments, Seq2SeqTrainer (Trainer API), Seq2SeqTrainingArguments are from https://github.com/huggingface/transformers
"""

import logging
import os
cur_file = os.path.realpath(__file__)

import re
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import is_main_process
from datasets import load_dataset, load_metric
from dataclasses import dataclass, field
from typing import Optional

from rouge_score import rouge_scorer


logger = logging.getLogger(__name__)

def evaluate_summary(reference,summary):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(reference,summary)
    # print(scores)
    return scores["rouge1"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task: str = field(
        default="summarization",
        metadata={
            "help": "The name of the task, should be summarization (or summarization_{dataset} for evaluating "
            "pegasus)"
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if not self.task.startswith("summarization"):
            raise ValueError(
                "`task` should be summarization or summarization_{dataset}"
            )
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

@dataclass
class DataValidationArguments:
    """
    Arguments pertaining to what parameters we are going to input to our model for validation.
    """
    min_summ_length: Optional[int] = field(
        default=100,
        metadata={
            "help": "The minimum length of the sequence to be generated."
        },
    )
    max_summ_length: Optional[int] = field(
        default=300,
        metadata={
            "help": "The maximum length of the sequence to be generated."
        },
    )

    num_beams: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of beams for beam search. 1 means no beam search."
        },
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Exponential penalty to the length. 1.0 means no penalty. Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences."
        },
    )

    no_repeat_ngram_size: Optional[int] = field(
        default=2,
        metadata={
            "help": " If set to int > 0, all ngrams of that size can only occur once."
        },
    )



summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():


    parser = HfArgumentParser((ModelArguments, DataTrainingArguments,DataValidationArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, test_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, test_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Get the default prefix if None is passed.
    if data_args.source_prefix is None:
        task_specific_params = model.config.task_specific_params
        if task_specific_params is not None:
            prefix = task_specific_params.get("prefix", "")
        else:
            prefix = ""
    else:
        prefix = data_args.source_prefix

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names


    # To serialize preprocess_function below, each of those four variables needs to be defined (even if we won't use
    # them all).
    text_column, summary_column = None, None

    if data_args.task.startswith("summarization"):
        # Get the column names for input/target.
        dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
        if data_args.text_column is None:
            text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            text_column = data_args.text_column
        if data_args.summary_column is None:
            summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            summary_column = data_args.summary_column

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    def preprocess_function(examples):

        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]

        # Tokenize Input
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if training_args.do_train:
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=label_pad_token_id)

    # Metric
    metric_name = "rouge" 
    metric = load_metric(metric_name)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]
        if metric_name == "sacrebleu":
            decoded_labels = [[label] for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)

        # Extract a few results from ROUGE
        if metric_name == "rouge":
            result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        else:
            result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:

        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval:

        model = trainer.model
        tokenizer = trainer.tokenizer
        print("\n")
        print("Running Evaluation Script")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        output = []
        inco = 0

        min_length = test_args.min_summ_length
        max_length = test_args.max_summ_length

        df_test = pd.read_csv(data_args.validation_file)

        for index,row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
            text = row['Text']
            ref = row['Summary']  
            input_tokenized = tokenizer.encode(text, return_tensors='pt',max_length=data_args.max_source_length, truncation=True).to(device)
            dim = list(input_tokenized.size())
            summary_ids = model.generate(input_tokenized,
                                                num_beams=test_args.num_beams,
                                                no_repeat_ngram_size=test_args.no_repeat_ngram_size,
                                                length_penalty=test_args.length_penalty,
                                                min_length=min_length,
                                                max_length=max_length,
                                                early_stopping=True)

            summ = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids][0]
            if summ.find("nnnn")!=-1:
                summ = re.sub(r'nnn*nn', '', summ)
                inco = inco + 1
            score = evaluate_summary(ref,summ)
            output.append((score.precision,ref,summ))
        
        print("Evaluation Completed")

        precision = [round(x[0],4) for x in output]
        fmeasure = [round(evaluate_summary(x[1],x[2]).fmeasure,4) for x in output]
        actual = [x[1] for x in output]
        generated = [re.sub(r'nnn*n', '',x[2]) for x in output]

        df = pd.DataFrame({'Generated Summary':generated,'Actual Summary':actual, 'Precision': precision, 'F Score': fmeasure})
        csv_output = os.path.join(training_args.output_dir, str(len(df_test)) +  "-test_results.csv")
        df.to_csv(csv_output)

        print("Evaluation results saved in {}".format(csv_output))

        output_df = pd.read_csv(csv_output)
        length_df = len(output_df)
        top = 10
        if length_df < 20:
            top = int(length_df/2) - 1
            if top <=0 :
                top = 1

        if length_df!=0:  
            final_output = [(x['Precision'],x['Actual Summary'],x['Generated Summary'],x['F Score']) for ind,x in output_df.iterrows()]

        output_desc = sorted(final_output, key = lambda x: -x[0])

        fsc = np.mean([t[3] for t in output_desc])
        pre = np.mean([t[0] for t in output_desc])

        output_eval_file = os.path.join(training_args.output_dir, "evaluation_scores.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                writer.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
                writer.write("\n")
                writer.write("Mean F Measure: {:.4f}".format(fsc))
                writer.write("\n")
                writer.write("Mean Precision (Rouge1): {:.4f}".format(pre))

                writer.write("\n")
                writer.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
                writer.write("\n")
                writer.write("Best {}: ".format(top))
                writer.write("\n")

                for tup in output_desc[0:top]:
                    writer.write("F Measure: {}".format(tup[3]))
                    writer.write("\n")
                    writer.write("Precision: {}".format(tup[0]))
                    writer.write("\n")
                    writer.write("Actual Summary:")
                    writer.write("\n")
                    writer.write(tup[1])
                    writer.write("\n")
                    writer.write("Generated Summary:")
                    writer.write("\n")
                    writer.write(tup[2])
                    writer.write("\n")
                    writer.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    writer.write("\n")
                writer.write("\n\n")

                writer.write("Worst {}: ".format(top))
                writer.write("\n")

                n = len(final_output)
                output_asc = sorted(final_output, key = lambda x: x[0])
                for tup in output_asc[:top]:
                    writer.write("F Measure: {}".format(tup[3]))
                    writer.write("\n")
                    writer.write("Precision: {}".format(tup[0]))
                    writer.write("\n")
                    writer.write("Actual Summary: ")
                    writer.write("\n")
                    writer.write(tup[1])
                    writer.write("\n")
                    writer.write("Generated Summary:")
                    writer.write("\n")
                    writer.write(tup[2])
                    writer.write("\n")
                    writer.write("-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
                    writer.write("\n")
    
        print("Evaluation scores saved in {}".format(output_eval_file))


    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
