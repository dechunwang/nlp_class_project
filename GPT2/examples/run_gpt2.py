#!/usr/bin/env python3

import argparse
import logging
from tqdm import tqdm, trange
import os
import torch
import torch.nn.functional as F
import numpy as np
import csv
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer, GPT2DoubleHeadsModel, OpenAIAdam, cached_path, WEIGHTS_NAME, CONFIG_NAME

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

import torch.nn as nn
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import pickle

ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def load_rocstories_dataset(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = csv.reader(f)
        output = []
        next(f) # skip the first line
        for line in f:
            #print(line[-1])
            output.append((' '.join(line[1:5]), line[5], line[6], int(line[-1])-1))
            #output.append((' '.join(line[1:5]), line[5]))

        #sys.exit("exit")

    return output


def load_rocstories_dataset_wo_neg(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = csv.reader(f)
        output = []
        next(f) # skip the first line
        for line in tqdm(f):
            #print(line[-1])
            output.append((' '.join(line[1:5]), line[5]))

        #sys.exit("exit")

    return output

def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
            with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
            #with_cont1 = story[:cap_length] + cont1[:cap_length]
            with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            #with_cont2 = story[:cap_length] + cont2[:cap_length]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1)-1] = with_cont1[1:]
            lm_labels[i, 1, :len(with_cont2)-1] = with_cont2[1:]
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets




def pre_process_datasets_wo_neg(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    n_batch = len(encoded_datasets)
    input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
    lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)

    for i, (story, cont1), in enumerate(encoded_datasets):
        with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        input_ids[i, :len(with_cont1)] = with_cont1
        lm_labels[i, :len(with_cont1) - 1] = with_cont1[1:]
    all_inputs = (input_ids, lm_labels)
    tensor_datasets.append(torch.tensor(all_inputs))

    return tensor_datasets


def top_k_logits(logits, k):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        return torch.where(logits < batch_mins, torch.ones_like(logits) * -1e10, logits)

def sample_sequence(model, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, device='cuda', sample=True):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
        context = torch.tensor(context, device=device, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
    prev = context
    output = context
    past = None
    with torch.no_grad():
        for i in trange(length):
            logits, past = model(prev, past=past)
            logits = logits[:, -1, :] / temperature
            logits = top_k_logits(logits, k=top_k)
            log_probs = F.softmax(logits, dim=-1)
            if sample:
                prev = torch.multinomial(log_probs, num_samples=1)
            else:
                _, prev = torch.topk(log_probs, k=1, dim=-1)
            output = torch.cat((output, prev), dim=1)
    return output

def run_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='gpt2', help='pretrained model name or path to local checkpoint')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nsamples", type=int, default=1)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.", default=True)
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.", default=True)
    parser.add_argument('--train_dataset', type=str, default='/home/chenxi/Desktop/roc/train/trains_large.csv')
    parser.add_argument("--output_dir",  type=str, default='../log_gpt2/',
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--eval_dataset', type=str, default='/home/chenxi/Desktop/roc/test/test.csv')
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument("--length", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument('--unconditional', action='store_true', help='If true, unconditional generation.')
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--n_valid', type=int, default=374)
    args = parser.parse_args()
    print(args)



    train_data_path = '/home/chenxi/Desktop/roc/train/train.csv'
    train_large_data_path = '/home/chenxi/Desktop/roc/train/trains_large.csv'
    eval_data_path = '/home/chenxi/Desktop/roc/test/test.csv'

    out_dir = '../log/'

    special_tokens = ['_start_', '_delimiter_', '_classify_']

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_ids = list(enc.convert_tokens_to_ids(token) for token in special_tokens)


    model = GPT2DoubleHeadsModel.from_pretrained(args.model_name_or_path)

    if args.do_eval:
        state = model.state_dict()
        state_dict = torch.load(args.output_dir + 'lmxl4.pth')
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict['model_state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params

        for k, v in new_state_dict.items():
            if k in state:
                state[k] = v

        model.load_state_dict(state)




    model.to(device)
    #model.eval()

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return enc.convert_tokens_to_ids(enc.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)






    if args.length == -1:
        args.length = model.config.n_ctx // 2
    elif args.length > model.config.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)



    if args.do_eval:
        train_dataset = load_rocstories_dataset(train_data_path)
        eval_dataset = load_rocstories_dataset(eval_data_path)
        datasets = (train_dataset, eval_dataset)
        encoded_datasets = tokenize_and_encode(datasets)
    else:
        train_dataset = load_rocstories_dataset_wo_neg(train_data_path)
        datasets = train_dataset


    if not args.do_eval and os.path.isfile('encode_dataset_gpt2.cache'):
        with open('encode_dataset_gpt2.cache', 'rb') as r:
            encoded_datasets = pickle.load(r)





    if not args.do_eval and not os.path.isfile('encode_dataset_gpt2.cache'):
        encoded_datasets = tokenize_and_encode(datasets)
        with open('encode_dataset_gpt2.cache', 'wb') as g:
            pickle.dump(encoded_datasets, g, pickle.HIGHEST_PROTOCOL)


    max_length = model.config.n_positions // 2 - 2




    if args.do_eval:
        input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) +3\
                       for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
    else:
        input_length = 0
        for story, cont1 in encoded_datasets:
            input_length = max(input_length, len(story[:max_length])+len(cont1[:max_length])+3)

    input_length = min(input_length, model.config.n_positions)



    if args.do_eval:
        tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
        train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]
        eval_data = TensorDataset(*eval_tensor_dataset)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        train_data = TensorDataset(*train_tensor_dataset)

    else:
        tensor_datasets = pre_process_datasets_wo_neg(encoded_datasets, input_length, max_length, *special_tokens_ids)
        train_tensor_dataset = tensor_datasets[0]
        train_data = TensorDataset(*train_tensor_dataset)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)



    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=args.learning_rate,
                           warmup=args.warmup_proportion,
                           max_grad_norm=args.max_grad_norm,
                           weight_decay=args.weight_decay,
                           t_total=num_train_optimization_steps)



    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None

        model = nn.DataParallel(model)
        model.train()
        ep = 0
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):

            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)

                if args.do_eval:
                    input_ids, mc_token_ids, lm_labels, mc_labels = batch
                    losses = model(input_ids=input_ids, mc_token_ids = mc_token_ids, lm_labels=lm_labels, mc_labels=mc_labels, eval=args.do_eval)
                    loss = args.lm_coef * losses[0] + losses[1]
                    #loss = losses[0]
                else:
                    input_ids, lm_labels = batch
                    losses = model(input_ids=input_ids, lm_labels=lm_labels, eval=args.do_eval)
                    loss = losses[0]

                loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

            if not args.do_eval:
                torch.save({
                    'model_state_dict': model.state_dict()
                }, args.output_dir + 'lmxl{}.pth'.format(ep))
                ep = ep+1

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.vocab_size = model_to_save.config.vocab_size+3
        model_to_save.config.to_json_file(output_config_file)
        enc.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = GPT2DoubleHeadsModel.from_pretrained(args.output_dir)
        #model = GPT2DoubleHeadsModel.from_pretrained(args.model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(args.output_dir)
        model.to(device)
        model = nn.DataParallel(model)

    if args.do_eval:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            with torch.no_grad():
                _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                _, mc_logits = model(input_ids, mc_token_ids)

            mc_logits = mc_logits.detach().cpu().numpy()
            mc_labels = mc_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

            eval_loss += mc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))




    # while True:
    #     context_tokens = []
    #     if not args.unconditional:
    #         raw_text = input("Model prompt >>> ")
    #         while not raw_text:
    #             print('Prompt should not be empty!')
    #             raw_text = input("Model prompt >>> ")
    #         context_tokens = enc.encode(raw_text)
    #         generated = 0
    #         for _ in range(args.nsamples // args.batch_size):
    #             out = sample_sequence(
    #                 model=model, length=args.length,
    #                 context=context_tokens,
    #                 start_token=None,
    #                 batch_size=args.batch_size,
    #                 temperature=args.temperature, top_k=args.top_k, device=device
    #             )
    #             out = out[:, len(context_tokens):].tolist()
    #             for i in range(args.batch_size):
    #                 generated += 1
    #                 text = enc.decode(out[i])
    #                 print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
    #                 print(text)
    #         print("=" * 80)
    #     if args.unconditional:
    #       generated = 0
    #       for _ in range(args.nsamples // args.batch_size):
    #           out = sample_sequence(
    #               model=model, length=args.length,
    #               context=None,
    #               start_token=enc.encoder['<|endoftext|>'],
    #               batch_size=args.batch_size,
    #               temperature=args.temperature, top_k=args.top_k, device=device
    #           )
    #           out = out[:,1:].tolist()
    #           for i in range(args.batch_size):
    #               generated += 1
    #               text = enc.decode(out[i])
    #               print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
    #               print(text)
    #       print("=" * 80)
    #       if args.unconditional:
    #           break

if __name__ == '__main__':
    run_model()


