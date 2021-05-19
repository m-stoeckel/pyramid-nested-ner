import torch

from pyramid_nested_ner.data.mutli_label_dataset import SentenceWindowMultiLabelNerDataset as Dataset
from pyramid_nested_ner.mutli_label_model import DocumentRNNSentenceWindowPyramid as Pyramid
from pyramid_nested_ner.training.multi_label_trainer import MultiLabelTrainer
from pyramid_nested_ner.training.optim import get_default_sgd_optim
from pyramid_nested_ner.utils.data import wrg_sentence_window_reader
from run_nne_sigmoid import get_default_argparser, load_vocab, write_report

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def instantiate_datasets(lexicon, args: dict):
    print("Loading data")
    train_data = list(wrg_sentence_window_reader("data/nne_raw/train/", args['sentence_window']))
    dev_data = list(wrg_sentence_window_reader("data/nne_raw/dev/", args['sentence_window']))
    test_data = list(wrg_sentence_window_reader("data/nne_raw/test/", args['sentence_window']))

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
        batch_size=args['eval_batch_size'],
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
        batch_size=args['eval_batch_size'],
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
    write_report(args, train_report, formatted_report)


if __name__ == '__main__':
    parser = get_default_argparser()

    # wrg_sentence_window_reader args
    parser.add_argument('--sentence_window', type=int, default=5)

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

    args = parser.parse_args()
    run_training(vars(args))
