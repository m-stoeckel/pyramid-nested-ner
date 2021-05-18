import argparse
import json
import zlib
from datetime import datetime

import torch

from pyramid_nested_ner.data.mutli_label_dataset import SigmoidMultiLabelNerDataset as Dataset
from pyramid_nested_ner.mutli_label_model import SigmoidMultiLabelPyramid as Pyramid
from pyramid_nested_ner.training.multi_label_trainer import MultiLabelTrainer
from pyramid_nested_ner.training.optim import get_default_sgd_optim
from pyramid_nested_ner.utils.data import wrg_reader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def crc32_hex_dict(mapping):
    return hex(zlib.crc32(json.dumps(mapping, sort_keys=True).encode()))[2:]


def timestamp():
    return datetime.now().strftime('%Y%m%d-%H_%M_%S')


def load_vocab(lex_size=400000):
    print("Loading vocabulary")
    with open(f"data/vocab_{lex_size}.list", encoding='utf-8') as fp:
        lexicon = [line.strip() for line in fp]

    return lexicon


def instantiate_datasets(lexicon, args: dict):
    print("Loading data")
    train_data = [data_point for data_point in wrg_reader("data/nne_concat/train.txt")]
    dev_data = [data_point for data_point in wrg_reader("data/nne_concat/dev.txt")]
    test_data = [data_point for data_point in wrg_reader("data/nne_concat/test.txt")]

    print("Generating entity lexicon")
    entities_lexicon = list(sorted({
        entity.name
        for data in (train_data, dev_data, test_data)
        for data_point in data
        for entity in data_point.entities
    }))
    print(entities_lexicon)

    if drop_entities := args.get("drop_entities"):
        for drop in drop_entities:
            if drop in entities_lexicon:
                entities_lexicon.remove(drop)

    print("Instantiating train dataset")
    train_dataloader = Dataset(
        train_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=args['shuffle_train'],
        batch_size=args['batch_size'],
        device=DEVICE,
        bucketing=args['bucketing']
    )
    print("Instantiating dev dataset")
    dev_dataloader = Dataset(
        dev_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=False,
        batch_size=args['batch_size'],
        device=DEVICE,
        bucketing=args['bucketing']
    )
    print("Instantiating test dataset")
    test_dataloader = Dataset(
        test_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=False,
        batch_size=args['batch_size'],
        device=DEVICE,
        bucketing=args['bucketing']
    )
    return train_dataloader, dev_dataloader, test_dataloader, entities_lexicon


def run_training(args: dict):
    word_lexicon = load_vocab(args['lex_size'])

    train_data, dev_data, test_data, entities_lexicon = instantiate_datasets(
        word_lexicon, args
    )

    print("Instantiating model")
    pyramid_ner = Pyramid(
        word_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        classifier_type=args['classifier_type'],
        word_embeddings=args['word_embeddings'],
        language_model=args['language_model'],
        char_embeddings_dim=args['char_embeddings_dim'],
        encoder_hidden_size=args['encoder_hidden_size'],
        encoder_output_size=args['encoder_output_size'],
        decoder_hidden_size=args['decoder_hidden_size'],
        inverse_pyramid=args['inverse_pyramid'],
        custom_tokenizer=args['custom_tokenizer'],
        pyramid_max_depth=args['pyramid_max_depth'],
        decoder_dropout=args['decoder_dropout'],
        encoder_dropout=args['encoder_dropout'],
        device=DEVICE
    )

    print(pyramid_ner.nnet)

    trainer = MultiLabelTrainer(pyramid_ner)
    optimizer, scheduler = get_default_sgd_optim(pyramid_ner.nnet.parameters())
    ner_model, train_report = trainer.train(
        train_data,
        optimizer=optimizer,
        scheduler=scheduler,
        dev_data=dev_data,
        epochs=args['epochs'],
        patience=args['patience'],
        grad_clip=args['grad_clip'],
        restore_weights_on=args['restore_weights_on']
    )
    train_report.plot_loss_report()
    train_report.plot_custom_report('micro_f1')

    formatted_report = trainer.test_model(test_data, out_dict=False)
    print(formatted_report)
    with open(f"report_{crc32_hex_dict(args)}_{timestamp()}.json", 'w') as fp:
        fp.write(json.dumps(
            {
                'model_name': Pyramid.__name__,
                'dataset_name': Dataset.__name__,
                'args': args,
                'train_report': train_report.report.to_dict(),
            }, indent=2
        ))
    with open(f"report_{crc32_hex_dict(args)}_{timestamp()}.tex", 'w') as fp:
        fp.write(formatted_report)


def get_default_argparser():
    parser = argparse.ArgumentParser(description='Run training')
    parser.add_argument('--word_embeddings', type=str, nargs='+', default=['en-glove', 'en-crawl'])
    parser.add_argument('--language_model', type=str, default=None)
    parser.add_argument('--classifier_type', type=str, default='linear',
                        choices=['linear', 'ovr_conv', 'ovr_multihead'])
    parser.add_argument('--restore_weights_on', type=str, default='loss')
    parser.add_argument('--shuffle_train', type=bool, default=False)
    parser.add_argument('--bucketing', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--char_embeddings_dim', type=int, default=60)
    parser.add_argument('--encoder_hidden_size', type=int, default=100)
    parser.add_argument('--encoder_output_size', type=int, default=200)
    parser.add_argument('--decoder_hidden_size', type=int, default=100)
    parser.add_argument('--pyramid_max_depth', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--inverse_pyramid', default=False)
    parser.add_argument('--custom_tokenizer', default=None)
    parser.add_argument('--decoder_dropout', type=float, default=0.4)
    parser.add_argument('--encoder_dropout', type=float, default=0.4)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--lex_size', type=int, default=400000, choices=(100000, 200000, 400000))
    parser.add_argument('--drop_entities', type=str, nargs='+', default=None)
    return parser


if __name__ == '__main__':
    parser = get_default_argparser()

    args = parser.parse_args()
    run_training(vars(args))
