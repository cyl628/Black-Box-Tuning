import datasets
from fastNLP import DataSet, Instance
from fastNLP.io import Loader, DataBundle
from functools import partial
from transformers import RobertaTokenizer
import os
import json
import pandas as pd
import csv


def convert_to_features(example_batch, tokenizer):
    input_encodings = tokenizer.batch_encode_plus(example_batch['input_text'])
    target_encodings = tokenizer.batch_encode_plus(example_batch['target_text'], add_special_tokens=False)
    mask_pos = []
    for input_ids in input_encodings['input_ids']:
        mask_pos.append(input_ids.index(tokenizer.mask_token_id))
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'mask_pos': mask_pos,
        'labels': target_encodings['input_ids'],
    }

    return encodings


class SST2Loader(Loader):
    dataset_project = {"train": "train",
                        "validation": "dev",
                        "test": "test"
                        }
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['sentence'], self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, f"{split}.tsv")
        examples = []
        with open(path, encoding='utf-8')as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                linelist = line.strip().split('\t')
                text_a = linelist[0]
                label = linelist[1]
                guid = "%s-%s" % (split, idx)
                example = {
                    "guid": guid,
                    "sentence": text_a,
                    "label": int(label)
                }
                # example = InputExample(guid=guid, text_a=text_a, label=self.get_label_id(label))
                examples.append(example)
        return examples

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('./glue', 'sst2', split=split)
        dataset = self.get_examples("../data/glue_data/SST-2", self.dataset_project[split])
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print('Example in {} set:'.format(split))
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class YelpPLoader(Loader):
    dataset_project = {"train": "train",
                        "validation": None,
                        "test": "test"
                        }
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "bad",
            1: "great",
        }
        self.tag2label = {
            "1": 0,
            "2": 1
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s . It was %s .' % (prompt, example['text'].replace("\\n", " "), self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s . It was %s .' % (example['text'].replace("\\n", " "), self.tokenizer.mask_token)
            example['target_text'] = self.label2text[example['label']]
        return example
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(self.dataset_project[split]))
        df = pd.read_csv(path, header=None)
        examples = []
        for idx, (label, text) in enumerate(zip(df[0], df[1])):
            text_a = text
            label = str(label)
            example = {
                "guid": str(idx),
                "text": text_a,
                "label": self.tag2label[label]
            }
            # example = InputExample(
            #         guid=str(idx), text_a=text_a, text_b="", label=label)
            examples.append(example)
        return examples

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('yelp_polarity', 'plain_text', split=split)
        dataset = self.get_examples("../data/Yelp", self.dataset_project[split])
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class AGNewsLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Tech"
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s News: %s' % (prompt, self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s News: %s' % (self.tokenizer.mask_token, example['text'])
            example['target_text'] = self.label2text[example['label']]
        return example
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                example = {
                    "guid": str(idx),
                    "text": text_b,
                    "title": text_a,
                    "label": int(label) - 1
                }
                # example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(label)-1)
                examples.append(example)
        return examples

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('ag_news', 'default', split=split)
        dataset = self.get_examples("../data/TextClassification/agnews", split)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class DBPediaLoader(Loader):
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Company",
            1: "Education",
            2: "Artist",
            3: "Athlete",
            4: "Office",
            5: "Transportation",
            6: "Building",
            7: "Natural",
            8: "Village",
            9: "Animal",
            10: "Plant",
            11: "Album",
            12: "Film",
            13: "Written",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s [ Category: %s ] %s' % (prompt, self.tokenizer.mask_token, example['content'].strip())
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '[ Category: %s ] %s' % (self.tokenizer.mask_token, example['content'].strip())
            example['target_text'] = self.label2text[example['label']]
        return example
    
    def get_examples(self, data_dir, split):
        examples = []
        label_file  = open(os.path.join(data_dir,"{}_labels.txt".format(split)),'r')
        labels  = [int(x.strip()) for x in label_file.readlines()]
        with open(os.path.join(data_dir,'{}.txt'.format(split)),'r') as fin:
            for idx, line in enumerate(fin):
                splited = line.strip().split(". ")
                text_a, text_b = splited[0], splited[1:]
                text_a = text_a+"."
                text_b = ". ".join(text_b)
                example = {
                    "guid": str(idx),
                    "title": text_a,
                    "content": text_b,
                    "label": int(labels[idx])
                }
                # example = InputExample(guid=str(idx), text_a=text_a, text_b=text_b, label=int(labels[idx]))
                examples.append(example)
        return examples

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('dbpedia_14', split=split)
        dataset = self.get_examples("../data/TextClassification/dbpedia", split)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class MRPCLoader(Loader):
    dataset_project = {"train": "msr_paraphrase_train",
                        "validation": "msr_paraphrase_test",
                        "test": "msr_paraphrase_test"
                        }
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "No",
            1: "Yes",
        }

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        return example
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.txt".format(self.dataset_project[split]))
        examples = []
        with open(path)as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                line_list = line.strip().split('\t')
                text_a = line_list[-2]
                text_b = line_list[-1]
                label = int(line_list[0])
                example = {
                    "guid": str(idx),
                    "sentence1": text_a, 
                    "sentence2": text_b,
                    "label": label
                }
                examples.append(example)
        return examples

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('./glue', 'mrpc', split=split)
        dataset = self.get_examples("../data/glue_data/MRPC", split)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class RTELoader(Loader):
    dataset_project = {"train": "train",
                        "validation": "dev",
                        "test": "test"
                        }
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "No",
        }
        self.tag2label = {
            "entailment": 0,
            "not_entailment": 1
        }
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.tsv".format(self.dataset_project[split]))
        examples = []
        with open(path)as f:
            lines = f.readlines()
            for idx, line in enumerate(lines[1:]):
                line_list = line.strip().split('\t')
                text_a = line_list[-3]
                text_b = line_list[-2]
                label = self.tag2label[line_list[-1]]
                example = {
                    "guid": str(idx),
                    "sentence1": text_a,
                    "sentence2": text_b,
                    "label": label
                }
                examples.append(example)
        return examples

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['sentence1'], self.tokenizer.mask_token, example['sentence2'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        # dataset = datasets.load_dataset('./glue', 'rte', split=split)
        dataset = self.get_examples("../data/glue_data/RTE" ,split=split)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle



class SNLILoader(Loader):
    dataset_project = {"train": "snli_1.0_train",
                        "dev": "snli_1.0_dev",
                        "test": "snli_1.0_test"
                        }
    def __init__(self, tokenizer=None, n_prompt_tokens=50):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
        self.n_prompt_tokens = n_prompt_tokens
        self.label2text = {
            0: "Yes",
            1: "Maybe",
            2: "No",
        }
        self.tag2label = {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2
        }
    
    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.jsonl".format(self.dataset_project[split]))
        examples = []
        with open(path)as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                data = json.loads(line)
                text_a = data["sentence1"]
                text_b = data["sentence2"]
                if data["gold_label"] not in self.tag2label:
                    continue
                label = self.tag2label[data["gold_label"]]
                example = {
                    "guid": str(idx),
                    "premise": text_a,
                    "hypothesis": text_b,
                    "label": label
                }
                examples.append(example)
        return examples

    def convert_examples(self, example):
        if self.n_prompt_tokens > 0:  # use randomly selected words as initial prompt
            offset = 1000
            prompt = self.tokenizer.decode(list(range(offset, offset + self.n_prompt_tokens)))
            example['input_text'] = '%s . %s ? %s , %s' % (prompt, example['premise'], self.tokenizer.mask_token ,example['hypothesis'])
            example['target_text'] = self.label2text[example['label']]
        else:
            example['input_text'] = '%s ? %s , %s' % (example['premise'], self.tokenizer.mask_token, example['hypothesis'])
            example['target_text'] = self.label2text[example['label']]
        return example

    def _load(self, split) -> DataSet:
        # load dataset with Huggingface's Datasets
        dataset = self.get_examples("../data/SNLI", split=split)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=dataset))
        # dataset = datasets.load_dataset('snli', split=split)
        dataset = dataset.filter(lambda example: example['label'] in [0, 1, 2])
        dataset = dataset.map(self.convert_examples, load_from_cache_file=False)
        print(dataset[0])
        dataset = dataset.map(partial(convert_to_features, tokenizer=self.tokenizer), batched=True, load_from_cache_file=False)
        # Convert to fastNLP.DataSet
        ds = DataSet()
        for ins in dataset:
            if len(ins["input_ids"]) <= 512:
                example = {
                    "input_ids": ins["input_ids"],
                    "attention_mask": ins["attention_mask"],
                    "mask_pos": ins["mask_pos"],
                    "labels": ins["labels"][0],
                }
                ds.append(Instance(**example))
        ds.set_input("input_ids", "attention_mask", "mask_pos")
        ds.set_target("labels")
        return ds

    def my_load(self, splits) -> DataBundle:
        datasets = {name: self._load(name) for name in splits}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle