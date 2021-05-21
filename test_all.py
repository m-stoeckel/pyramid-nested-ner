import itertools

import torch

from pyramid_nested_ner.data.mutli_label_dataset import SentenceWindowMultiLabelNerDataset, SigmoidMultiLabelNerDataset, \
    TokenWindowMultiLabelNerDataset
from pyramid_nested_ner.mutli_label_model import DocumentRNNSentenceWindowPyramid, PooledSentenceTransformerPyramid, \
    SigmoidMultiLabelPyramid
from pyramid_nested_ner.training.multi_label_trainer import MultiLabelTrainer
from pyramid_nested_ner.training.optim import get_default_sgd_optim
from pyramid_nested_ner.utils.data import wrg_reader, wrg_sentence_window_reader, wrg_token_window_reader
from run_nne_sigmoid import get_default_argparser, load_vocab

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_data_loader(train_dataset, dev_dataset, test_dataset, args):
    train_dataloader = train_dataset.get_dataloader(
        shuffle=args['shuffle_train'],
        batch_size=args['batch_size'],
        device=DEVICE,
        bucketing=args['bucketing']
    )
    dev_dataloader = dev_dataset.get_dataloader(
        shuffle=False,
        batch_size=args['eval_batch_size'],
        device=DEVICE,
        bucketing=args['bucketing']
    )
    test_dataloader = test_dataset.get_dataloader(
        shuffle=False,
        batch_size=args['eval_batch_size'],
        device=DEVICE,
        bucketing=args['bucketing']
    )
    return dev_dataloader, train_dataloader, test_dataloader


def run_sigmoid_test(args: dict):
    word_lexicon = load_vocab(args['lex_size'])

    print("Loading data")
    train_data = list(itertools.islice(wrg_reader("data/nne_concat/train.txt"), 64))
    dev_data = list(itertools.islice(wrg_reader("data/nne_concat/dev.txt"), 64))
    test_data = list(itertools.islice(wrg_reader("data/nne_concat/test.txt"), 64))

    print("Generating entity lexicon")
    entities_lexicon = list(sorted({
        entity.name
        for data in (train_data, dev_data, test_data)
        for data_point in data
        for entity in data_point.entities
    }))
    print(entities_lexicon)

    print("Instantiating train dataset")
    train_dataset = SigmoidMultiLabelNerDataset(
        train_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )
    print("Instantiating dev dataset")
    dev_dataset = SigmoidMultiLabelNerDataset(
        dev_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )
    print("Instantiating test dataset")
    test_dataset = SigmoidMultiLabelNerDataset(
        test_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )

    dev_dataloader, train_dataloader, test_dataloader = get_data_loader(train_dataset, dev_dataset, test_dataset, args)

    print("Instantiating SigmoidMultiLabelPyramid")
    pyramid_ner = SigmoidMultiLabelPyramid(
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
    run_training(pyramid_ner, train_dataloader, dev_dataloader, test_dataloader, args)


def run_sentence_window_test(args: dict):
    word_lexicon = load_vocab(args['lex_size'])

    print("Loading data")
    train_data = list(itertools.islice(wrg_sentence_window_reader("data/nne_raw/train/", args['sentence_window']), 64))
    dev_data = list(itertools.islice(wrg_sentence_window_reader("data/nne_raw/dev/", args['sentence_window']), 64))
    test_data = list(itertools.islice(wrg_sentence_window_reader("data/nne_raw/test/", args['sentence_window']), 64))

    print("Generating entity lexicon")
    entities_lexicon = list(sorted({
        entity.name
        for data in (train_data, dev_data, test_data)
        for data_point in data
        for entity in data_point.entities
    }))
    print(entities_lexicon)

    print("Instantiating train dataset")
    train_dataset = SentenceWindowMultiLabelNerDataset(
        train_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )
    print("Instantiating dev dataset")
    dev_dataset = SentenceWindowMultiLabelNerDataset(
        dev_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )
    print("Instantiating test dataset")
    test_dataset = SentenceWindowMultiLabelNerDataset(
        test_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )

    dev_dataloader, train_dataloader, test_dataloader = get_data_loader(train_dataset, dev_dataset, test_dataset, args)
    test_document_rnn(word_lexicon, train_dataloader, dev_dataloader, test_dataloader, entities_lexicon, args)

    dev_dataloader, train_dataloader, test_dataloader = get_data_loader(train_dataset, dev_dataset, test_dataset, args)
    test_sentence_transformer(word_lexicon, train_dataloader, dev_dataloader, test_dataloader, entities_lexicon, args)


def run_token_window_test(args: dict):
    word_lexicon = load_vocab(args['lex_size'])

    print("Loading data")
    train_data = list(itertools.islice(wrg_token_window_reader("data/nne_raw/train/", args['token_window']), 64))
    dev_data = list(itertools.islice(wrg_token_window_reader("data/nne_raw/dev/", args['token_window']), 64))
    test_data = list(itertools.islice(wrg_token_window_reader("data/nne_raw/test/", args['token_window']), 64))

    print("Generating entity lexicon")
    entities_lexicon = list(sorted({
        entity.name
        for data in (train_data, dev_data, test_data)
        for data_point in data
        for entity in data_point.entities
    }))
    print(entities_lexicon)

    print("Instantiating train dataset")
    train_dataset = TokenWindowMultiLabelNerDataset(
        train_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )
    print("Instantiating dev dataset")
    dev_dataset = TokenWindowMultiLabelNerDataset(
        dev_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )
    print("Instantiating test dataset")
    test_dataset = TokenWindowMultiLabelNerDataset(
        test_data,
        pyramid_max_depth=args['pyramid_max_depth'],
        token_lexicon=word_lexicon,
        entities_lexicon=entities_lexicon,
        custom_tokenizer=None,
        char_vectorizer=True,
    )

    dev_dataloader, train_dataloader, test_dataloader = get_data_loader(train_dataset, dev_dataset, test_dataset, args)
    test_document_rnn(word_lexicon, train_dataloader, dev_dataloader, test_dataloader, entities_lexicon, args)

    dev_dataloader, train_dataloader, test_dataloader = get_data_loader(train_dataset, dev_dataset, test_dataset, args)
    test_sentence_transformer(word_lexicon, train_dataloader, dev_dataloader, test_dataloader, entities_lexicon, args)


def test_document_rnn(word_lexicon, train_dataloader, dev_dataloader, test_dataloader, entities_lexicon, args):
    print("Instantiating DocumentRNNSentenceWindowPyramid")
    pyramid_ner = DocumentRNNSentenceWindowPyramid(
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
        use_pre=args['use_pre'],
        use_post=args['use_post'],
        hidden_size=args['hidden_size'],
        rnn_layers=args['rnn_layers'],
        reproject_words=args['reproject_words'],
        reproject_words_dimension=args['reproject_words_dimension'],
        bidirectional=args['bidirectional'],
        dropout=args['dropout'],
        word_dropout=args['word_dropout'],
        locked_dropout=args['locked_dropout'],
        device=DEVICE
    )

    print(pyramid_ner.nnet)
    run_training(pyramid_ner, train_dataloader, dev_dataloader, test_dataloader, args)


def test_sentence_transformer(word_lexicon, train_dataloader, dev_dataloader, test_dataloader, entities_lexicon, args):
    print("Instantiating PooledSentenceTransformerPyramid")
    pyramid_ner = PooledSentenceTransformerPyramid(
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
        model=args['model'],
        batch_size=args['transformer_batch_size'],
        embedding_encoder_type=args['transformer_embedding_encoder_type'],
        embedding_encoder_hidden_size=args['transformer_embedding_encoder_hidden_size'],
        encoder_type=args['encoder_type'],
        transformer_encoder_output_size=args['transformer_encoder_output_size'],
        dropout=args['encoder_dropout'],
        padding_idx=args['padding_idx'],
        casing=args['casing'],
        use_pre=args['use_pre'],
        use_post=args['use_post'],
        device=DEVICE
    )

    print(pyramid_ner.nnet)
    run_training(pyramid_ner, train_dataloader, dev_dataloader, test_dataloader, args)


def run_training(pyramid_ner, train_dataloader, dev_dataloader, test_dataloader, args):
    trainer = MultiLabelTrainer(pyramid_ner)
    optimizer, scheduler = get_default_sgd_optim(pyramid_ner.nnet.parameters())
    ner_model, train_report = trainer.train(
        train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        dev_data=dev_dataloader,
        epochs=args['epochs'],
        patience=args['patience'],
        grad_clip=args['grad_clip'],
        restore_weights_on=args['restore_weights_on']
    )
    train_report.plot_loss_report()
    train_report.plot_custom_report('micro_f1')
    formatted_report = trainer.test_model(test_dataloader, out_dict=False)
    print("\n".join(formatted_report.split("\n")[0:8]))


if __name__ == '__main__':
    parser = get_default_argparser()

    # wrg_sentence_window_reader args
    parser.add_argument('--sentence_window', type=int, default=5)
    parser.add_argument('--token_window', type=int, default=64)

    # DocumentRNN args
    parser.add_argument('--use_pre', const=True, action='store_const', default=True)
    parser.add_argument('--use_post', const=True, action='store_const', default=False)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--rnn_layers', type=int, default=1)
    parser.add_argument('--reproject_words', const=True, action='store_const', default=True)
    parser.add_argument('--reproject_words_dimension', type=int, default=None)
    parser.add_argument('--bidirectional', const=True, action='store_const', default=False)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--word_dropout', type=float, default=0.0)
    parser.add_argument('--locked_dropout', type=float, default=0.0)

    # PooledSentenceTransformerPyramid args
    parser.add_argument('--model', type=str, default="paraphrase-distilroberta-base-v1")
    parser.add_argument('--transformer_batch_size', type=int, default=1)
    parser.add_argument('--transformer_embedding_encoder_type', default='rnn')
    parser.add_argument('--transformer_embedding_encoder_hidden_size', default=128)
    parser.add_argument('--encoder_type', type=str, default='identity')
    parser.add_argument('--transformer_encoder_output_size', default=64)
    parser.add_argument('--padding_idx', default=0)
    parser.add_argument('--casing', const=True, action='store_const', default=True)

    args = parser.parse_args()

    run_sigmoid_test(vars(args))
    run_sentence_window_test(vars(args))
    run_token_window_test(vars(args))
