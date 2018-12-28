from data_util import Vocab


def main():
    train_query_file = 'data/train.query.txt'
    train_context_file = 'data/train.context.txt'
    train_tables_file = 'data/train.tables.txt'
    train_answer_file = 'data/train.answer.txt'
    vocab_file = "data/vocab"
    
    embedding_file = "data/glove.npz"
    glove_file = "data/glove.840B.300d.txt"
    dict_file = "data/dict.p"
    max_vocab_size = 5e4
    Vocab.build_vocab(train_query_file, train_context_file, train_tables_file, train_answer_file, 
                                            vocab_file, dict_file, glove_file, embedding_file, max_vocab_size)


if __name__ == "__main__":
    main()
