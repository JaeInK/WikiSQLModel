from data_util import batch_loader, Vocab, load_data, zero_padding, max_len
from jjmodel import Wiki
from config import Config
import numpy as np

def main():
    config = Config()
    vocab = Vocab(config.dict_file)
    tq, tc, q, c, a_s, a_t = load_data(config.train_query_file, config.train_context_file, config.train_tables_file, config.train_answer_file, vocab, config.debug)
    #dev_q, dev_c, dev_s, dev_a_s, dev_a_t = load_data(config.dev_query_file, config.dev_context_file, config.dev_tables_file, config.dev_answer_file, vocab, config.debug)
    max_q_len = max_len(q)
    max_c_len = max_len(c)
    max_a_len = max_len(a_s)
    train_data = list(zip(tq, tc, q, c, a_s, a_t))
    #dev_data = list(zip(dev_q, dev_c, dev_s, dev_a_s, dev_a_t))
    wiki = Wiki(config)
    wiki.build_model()
    best_score = 0
    print("start")
    for i in range(config.num_epochs):
        epoch = i + 1
        print(epoch)
        batches = batch_loader(train_data, config.batch_size, shuffle=False)
        for batch in batches:
            batch_tq, batch_tc, batch_q, batch_c, batch_a_s, batch_a_t = zip(*batch)
            question_lengths, padded_q = zero_padding(batch_q, max_q_len)
            context_lengths, padded_c = zero_padding(batch_c, max_c_len)
            answer_start_lengths, padded_a_s = zero_padding(batch_a_s, max_a_len)
            # loss, acc, pred, step = wiki.train(padded_q, question_lengths, padded_c, context_lengths,
            #                                     padded_a_s, answer_start_lengths, config.dropout)
            loss,  step = wiki.train(padded_q, question_lengths, padded_c, context_lengths,
                                                padded_a_s, batch_a_t, answer_start_lengths, max_q_len, 
                                                 max_c_len, max_a_len, config.dropout)
            # train_batch_acc, train_batch_em, train_batch_loss = wiki.eval(padded_q, question_lengths, padded_c,
            #                                                                context_lengths,
            #                                                                padded_s, sequence_lengths,
            #                                                                sentence_lengths, batch_s_idx,
            #                                                                batch_ans, batch_spans)
            train_batch_loss = wiki.eval(padded_q, question_lengths, padded_c, context_lengths, 
                                                        padded_a_s, batch_a_t, answer_start_lengths, max_q_len, 
                                                        max_c_len, max_a_len, config.dropout)
            print(loss)
            print(train_batch_loss)
            print("done")


            # dev_em = np.mean(total_em)
            # dev_acc = np.mean(total_acc)
            # dev_loss = np.mean(total_loss)
            # wiki.write_summary(dev_acc, dev_em, dev_loss, mode="dev")
            # wiki.write_summary(train_batch_acc, train_batch_em, train_batch_loss, mode="train")
            # print("after %d step, dev_em:%.2f" % (step, dev_em))
            # if dev_em > best_score:
            #     best_score = dev_em
            #     print("new score! em: %.2f, acc:%.2f" % (dev_em, dev_acc))
            #     wiki.save_session(config.dir_model)
            

    print("end")


if __name__ == "__main__":
    main()
