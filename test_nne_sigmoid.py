import numpy as np
import torch

from pyramid_nested_ner.data.mutli_label_dataset import SigmoidMultiLabelNerDataset
from pyramid_nested_ner.mutli_label_model import SigmoidMultiLabelPyramidNer
from pyramid_nested_ner.training.multi_label_trainer import MultiLabelTrainer
from pyramid_nested_ner.training.optim import get_default_sgd_optim
from pyramid_nested_ner.utils.data import wrg_reader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DROP_ENTITIES = (
)


def get_default_model(lexicon, entities_lexicon):
    return SigmoidMultiLabelPyramidNer(
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
    return SigmoidMultiLabelPyramidNer(
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


def load_data(lex_size=1):
    print("Loading vocabulary")
    lex_size = (100000, 200000, 400000)[lex_size]
    with open(f"data/vocab_{lex_size}.list", encoding='utf-8') as fp:
        lexicon = [line.strip() for line in fp]

    return lexicon


def instantiate_datasets(lexicon, dataset_size=(1024, 256, 256)):
    print("Loading data")  # FIXME
    train_data = [data_point for data_point in wrg_reader("data/nne_concat/train.txt")][:dataset_size[0]]
    dev_data = [data_point for data_point in wrg_reader("data/nne_concat/dev.txt")][:dataset_size[1]]
    test_data = [data_point for data_point in wrg_reader("data/nne_concat/test.txt")][:dataset_size[2]]

    print("Generating entity lexicon")
    entities_lexicon = list(sorted({
        entity.name
        for data in (train_data, dev_data, test_data)
        for data_point in data
        for entity in data_point.entities
    }))
    print(entities_lexicon)

    for drop in DROP_ENTITIES:
        if drop in entities_lexicon:
            entities_lexicon.remove(drop)

    print("Instantiating train dataset")
    train_dataloader = SigmoidMultiLabelNerDataset(
        train_data,
        pyramid_max_depth=8,
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=False,
        batch_size=128,
        device=DEVICE,
        bucketing=True
    )
    print("Instantiating dev dataset")
    dev_dataloader = SigmoidMultiLabelNerDataset(
        dev_data,
        pyramid_max_depth=8,
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=False,
        batch_size=64,
        device=DEVICE,
        bucketing=True
    )
    print("Instantiating test dataset")
    test_dataloader = SigmoidMultiLabelNerDataset(
        test_data,
        pyramid_max_depth=8,
        token_lexicon=lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    ).get_dataloader(
        shuffle=False,
        batch_size=64,
        device=DEVICE,
        bucketing=True
    )
    return train_dataloader, dev_dataloader, test_dataloader, entities_lexicon


def run_training(lex_size=2, use_bert=False):
    lexicon = load_data(lex_size=lex_size)

    train_data, dev_data, test_data, entities_lexicon = instantiate_datasets(
        lexicon
    )

    print("Instantiating model")
    if use_bert:
        pyramid_ner = get_bert_model(lexicon, entities_lexicon)
    else:
        pyramid_ner = get_default_model(lexicon, entities_lexicon)

    print(pyramid_ner.nnet)

    # trainer = PyramidNerTrainer(pyramid_ner)
    trainer = MultiLabelTrainer(pyramid_ner)
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
