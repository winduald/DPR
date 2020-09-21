from dpr.data.index import BM25Index
import json


def BM25_retrieval(test_filename):
    with open(test_filename, 'r', encoding='utf-8') as fin:
        all_data = json.load(fin)
    all_posts = []
    all_ids = []

    #get all negatives
    for example in all_data:
        if len(example['positive_ctxs']) == 0:
            continue
        for tag in ['negative_ctxs', 'positive_ctxs', 'hard_negative_ctxs']:
            for ctx in example[tag]:
                all_posts.append(ctx['title'] + ' ' + ctx['text'])
                all_ids.append(ctx['passage_id'])

    bm25_index = BM25Index(all_posts, all_ids)
    num_ctx = len(all_posts)

    #calculate BM25 average rank
    num_pos = 0
    total_rank = 0
    for example in all_data:
        question = example['question']
        res_ids, res_texts, scores = bm25_index.retreival_ranked_list(question, num_ctx)
        # print(question)
        # print(res_texts[0: 20])
        # exit(1)
        for pos_ctx in example['positive_ctxs']:
            post_ctx_text, post_ctx_id = pos_ctx['title'] + ' ' + pos_ctx['text'], pos_ctx['passage_id']
            i = res_ids.index(post_ctx_id)
            assert res_texts[i] == post_ctx_text
            total_rank += i
            num_pos += 1
            # print(question)
            # print(post_ctx_text)
            # print(res_texts[0:5])
            # exit(1)
            # print(i)
    
    print("average rank is %f" %(float(total_rank)/float(num_pos)))
    print("number of example %d" %num_pos)


def evaluate_BM25():
    test_filename = '/checkpoint/xiaojianwu/data/semeval/dpr/semeval_q1bs_q2bs_test.json'
    test_filename_askubuntu = '/checkpoint/xiaojianwu/data/askubuntu/dpr/askubuntu_q1bs_q2bs_test.json'
    BM25_retrieval(test_filename_askubuntu)

if __name__ == "__main__":
    evaluate_BM25()