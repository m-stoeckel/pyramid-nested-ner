import json

import torch

from pyramid_nested_ner.data.mutli_label_dataset import SentenceWindowsMultiLabelNerDataset as Dataset
from pyramid_nested_ner.mutli_label_model import SentenceTransformerPyramid as Pyramid
from pyramid_nested_ner.training.multi_label_trainer import MultiLabelTrainer
from pyramid_nested_ner.training.optim import get_default_sgd_optim
from run_nne_document_rnn import instantiate_datasets
from run_nne_sigmoid import crc32_hex_dict, get_default_argparser, load_vocab, timestamp

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
        model=args['model'],
        batch_size=args['transformer_batch_size'],
        embedding_encoder_type=args['transformer_embedding_encoder_type'],
        embedding_encoder_hidden_size=args['transformer_embedding_encoder_hidden_size'],
        encoder_type=args['encoder_type'],
        transformer_encoder_output_size=args['transformer_encoder_output_size'],
        padding_idx=args['padding_idx'],
        casing=args['casing'],
        use_pre=args['use_pre'],
        use_post=args['use_post'],
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
                'formatted_report': formatted_report,
            }
        ))


if __name__ == '__main__':
    parser = get_default_argparser()

    # wrg_sentence_window_reader args
    parser.add_argument('--sentence_window', type=int, default=5)

    # SentenceTransformerPyramid args
    parser.add_argument('--model', type=str, default="paraphrase-distilroberta-base-v1")
    parser.add_argument('--transformer_batch_size', type=int, default=1)
    parser.add_argument('--transformer_embedding_encoder_type', default='rnn')
    parser.add_argument('--transformer_embedding_encoder_hidden_size', default=128)
    parser.add_argument('--encoder_type', type=str, default='identity')
    parser.add_argument('--transformer_encoder_output_size', default=64)
    parser.add_argument('--padding_idx', default=0)
    parser.add_argument('--casing', const=True, action='store_const', default=True)
    parser.add_argument('--use_pre', const=True, action='store_const', default=True)
    parser.add_argument('--use_post', const=True, action='store_const', default=False)

    args = parser.parse_args()
    run_training(vars(args))
