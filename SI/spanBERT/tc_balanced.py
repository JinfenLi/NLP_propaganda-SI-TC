"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsampler import ImbalancedDatasetSampler

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


PRED_FILE = "tc_predictions.tsv"
EVAL_FILE = "eval_results.txt"
TEST_FILE = "test_results.txt"

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class TC_Processor():
    """Processor for the CoLA data set (GLUE version)."""

    def _read_tsv(cls, input_file):
        """Reads a tab separated value file."""
        import pandas as pd
        lines = pd.read_csv(input_file,sep='\t',header=None).values

        return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(14)]


    def _create_examples(self, lines, set_type):
        examples = []

        for (i, line) in enumerate(lines):


            guid = "%s-%s" % (set_type, i)
            if set_type == 'test':
                text_a = line[1]
                label = None
            else:
                text_a = line[1]
                label = str(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples






def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Loads a data file into a list of `InputBatch`s."""




    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if example.label is None:
            label_id = None
        else:
            label_map = {label: i for i, label in enumerate(label_list)}
            label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if example.label is None:
                logger.info("label: <UNK>")
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def evaluate(model, device, eval_dataloader):
    model.eval()
    preds = []
    for eval_example in eval_dataloader:
        if len(eval_example) == 4:
            input_ids, input_mask, segment_ids, label_ids = eval_example
        else:
            input_ids, input_mask, segment_ids = eval_example

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)

        with torch.no_grad():
            logits = model(input_ids, segment_ids, input_mask, labels=None)


        if len(preds) == 0:
            preds.append(logits.detach().cpu().numpy())
        else:
            preds[0] = np.append(
                preds[0], logits.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    preds = np.argmax(preds, axis=1)


    return preds

def call_get_label(dataset, idx):
    # dataset.imgs[idx][1]
    print(dataset[idx][3])

    return dataset[idx][3]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.do_train:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(os.path.join(args.output_dir, "eval.log"), 'w'))
    logger.info(args)



    processor = TC_Processor()
    label_list = processor.get_labels()
    id2label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.model, do_lower_case=args.do_lower_case)


    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir)
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        if args.train_mode == 'sorted' or args.train_mode == 'random_sorted':
            train_features = sorted(train_features, key=lambda f: np.sum(f.input_mask))
        else:
            random.shuffle(train_features)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        if args.fp16:
            all_label_ids = all_label_ids.half()


        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, drop_last=True)
        # print(type(call_get_label(train_data,3)))
        train_dataloader = DataLoader(
            train_data,
            sampler=ImbalancedDatasetSampler(train_data),
            batch_size=args.train_batch_size,
            drop_last=True
        )
        train_batches = [batch for batch in train_dataloader]
        eval_step = max(1, len(train_batches) // args.eval_per_epoch)

        num_train_optimization_steps = \
            len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

        logger.info("***** Training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

        best_result = None
        # lrs = [args.learning_rate] if args.learning_rate else \
        #     [1e-6, 2e-6, 3e-6, 5e-6, 1e-5, 2e-5, 3e-5, 5e-5]
        lr = 1e-6
        # for lr in lrs:
        cache_dir = args.cache_dir if args.cache_dir else \
            PYTORCH_PRETRAINED_BERT_CACHE
        model = BertForSequenceClassification.from_pretrained(
            args.model, cache_dir=cache_dir, num_labels=num_labels)
        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        if args.fp16:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex"
                                  "to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                  lr=lr,
                                  bias_correction=False,
                                  max_grad_norm=1.0)
            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=lr,
                                 warmup=args.warmup_proportion,
                                 t_total=num_train_optimization_steps)

        global_step = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        tr_loss = 0
        start_time = time.time()

        for epoch in range(int(args.num_train_epochs)):
            model.train()
            logger.info("Start epoch #{} (lr = {})...".format(epoch, lr))
            if args.train_mode == 'random' or args.train_mode == 'random_sorted':
                random.shuffle(train_batches)
            for step, batch in enumerate(train_batches):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, segment_ids, input_mask, labels=None)
                # weights = [0.4, 1]
                # class_weights = torch.FloatTensor(weights).cuda()
                weight_CE = torch.DoubleTensor([2.3,4.8,1.2,1.8,3.4,8,7.6,3.7,34.6,17.3,10.1,2.1,1.2,1.8])
                loss_fct = CrossEntropyLoss(weight=weight_CE)
                loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))


                if n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        lr_this_step = lr * \
                            warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (step + 1) % eval_step == 0:
                    logger.info('Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}'.format(
                                 epoch, step + 1, len(train_dataloader),
                                 time.time() - start_time, tr_loss / nb_tr_steps))

            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(args.output_dir, 'epoch'+str(epoch)+'_'+WEIGHTS_NAME)
            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
            model_to_save.config.to_json_file(output_config_file)
            tokenizer.save_vocabulary(args.output_dir)


    if args.do_test:

        eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Test *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

        model = BertForSequenceClassification.from_pretrained(args.output_dir, num_labels=num_labels)
        if args.fp16:
            model.half()
        model.to(device)
        preds = evaluate( model, device, eval_dataloader)
        pred_file = os.path.join(args.output_dir, PRED_FILE)
        with open(pred_file, "w") as f_out:
            f_out.write("index\tprediction\n")
            for i, pred in enumerate(preds):
                f_out.write("%d\t%s\n" % (i, id2label[pred]))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model", default=None, type=str, required=True)
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--eval_per_epoch", default=10, type=int,
                        help="How many times it evaluates on dev set per epoch")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_mode", type=str, default='random_sorted', choices=['random', 'sorted', 'random_sorted'])
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_test", action='store_true', help="Whether to eval on the test set.")
    parser.add_argument("--eval_set", type=str, default=None, help="Whether to evalu on the test set.")
    parser.add_argument("--train_batch_size", default=4, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=None, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()

    main(args)