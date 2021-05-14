import itertools
import json

import numpy as np
import torch

from pyramid_nested_ner.data.dataset import PyramidNerDataset
from pyramid_nested_ner.model import PyramidNer
from pyramid_nested_ner.training.optim import get_default_sgd_optim
from pyramid_nested_ner.training.trainer import PyramidNerTrainer
from pyramid_nested_ner.utils.data import jsonline_data_reader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_wordnet_split(split='train', limit=-1):
    with open(f"data/wordnet-dataset.{split}.jsonl", 'r') as fp:
        return [json.loads(line) for line in itertools.islice(fp, limit)]


def get_default_model(lexicon, entities_lexicon):
    return PyramidNer(
        word_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        word_embeddings=['en-glove', 'en-crawl'],
        language_model=None,
        char_embeddings_dim=60,
        encoder_hidden_size=100,
        encoder_output_size=200,
        decoder_hidden_size=100,
        inverse_pyramid=False,
        custom_tokenizer=None,
        pyramid_max_depth=8,
        decoder_dropout=0.4,
        encoder_dropout=0.4,
        device=DEVICE,
    )


def get_bert_model(lexicon, entities_lexicon):
    return PyramidNer(
        word_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        word_embeddings=['en-glove', 'en-crawl'],
        language_model='bert-base-multilingual-cased',
        char_embeddings_dim=60,
        encoder_hidden_size=100,
        encoder_output_size=200,
        decoder_hidden_size=100,
        inverse_pyramid=False,
        custom_tokenizer=None,
        pyramid_max_depth=8,
        decoder_dropout=0.4,
        encoder_dropout=0.4,
        device=DEVICE,
    )


def load_data(dataset_size, lex_size=1):
    print("Loading train data")
    train = load_wordnet_split('train', dataset_size[0])

    print("Loading test & dev data")
    test, dev = load_wordnet_split('test', dataset_size[1]), load_wordnet_split('dev', dataset_size[2])

    entities_lexicon = {entity['category'] for sample in train for entity in sample['entities']}
    entities_lexicon |= {entity['category'] for sample in test for entity in sample['entities']}
    entities_lexicon |= {entity['category'] for sample in dev for entity in sample['entities']}
    entities_lexicon = list(entities_lexicon)

    print("Loading vocabulary")
    lex_size = (100000, 200000, 400000)[lex_size]
    with open(f"data/vocab_{lex_size}.list", encoding='utf-8') as fp:
        lexicon = [line.strip() for line in fp]

    return lexicon, entities_lexicon


def instantiate_datasets(lexicon, entities_lexicon, dataset_size, drop_entities=None):
    print("Instantiating train dataset")
    train_data = PyramidNerDataset(
        jsonline_data_reader('data/wordnet-dataset.train.jsonl', dataset_size[0], drop_entities),
        pyramid_max_depth=8,
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=True,
        batch_size=128,
        device=DEVICE,
        bucketing=True
    )
    print("Instantiating test dataset")
    test_data = PyramidNerDataset(
        jsonline_data_reader('data/wordnet-dataset.test.jsonl', dataset_size[1], drop_entities),
        pyramid_max_depth=8,
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=True,
        batch_size=32,
        device=DEVICE,
        bucketing=True
    )
    print("Instantiating dev dataset")
    dev_data = PyramidNerDataset(
        jsonline_data_reader('data/wordnet-dataset.dev.jsonl', dataset_size[2], drop_entities),
        pyramid_max_depth=8,
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=True,
        batch_size=32,
        device=DEVICE,
        bucketing=True
    )
    return dev_data, test_data, train_data


def run_training(lex_size=2, use_bert=False):
    dataset_size = (40000, 5000, 2000)
    lexicon, entities_lexicon = load_data(dataset_size, lex_size=lex_size)

    drop_entities = (
        'Tops',
        # 'act',
        # 'animal',
        # 'artifact',
        'attribute',
        # 'body',
        # 'cognition',
        # 'communication',
        'event',
        'feeling',
        # 'food',
        # 'group',
        # 'location',
        # 'object',
        # 'person',
        # 'phenomenon',
        # 'plant',
        'possession',
        'process',
        # 'quantity',
        'relation',
        'shape',
        # 'state',
        # 'substance',
        # 'time'
    )
    for drop in drop_entities:
        if drop in entities_lexicon:
            entities_lexicon.remove(drop)

    dev_data, test_data, train_data = instantiate_datasets(
        lexicon,
        entities_lexicon,
        dataset_size,
        drop_entities=drop_entities
    )

    print("Instantiating model")
    if use_bert:
        pyramid_ner = get_bert_model(lexicon, entities_lexicon)
    else:
        pyramid_ner = get_default_model(lexicon, entities_lexicon)

    print(pyramid_ner.nnet)

    trainer = PyramidNerTrainer(pyramid_ner)
    optimizer, scheduler = get_default_sgd_optim(pyramid_ner.nnet.parameters())
    ner_model, report = trainer.train(
        train_data,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_weights_on='loss',
        epochs=30,
        dev_data=dev_data,
        patience=np.inf,
        grad_clip=5.0
    )
    report.plot_loss_report()
    report.plot_custom_report('micro_f1')
    print(trainer.test_model(test_data, out_dict=False))


if __name__ == '__main__':
    run_training()
