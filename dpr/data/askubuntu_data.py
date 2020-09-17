import xml.etree.ElementTree as ET
from collections import defaultdict
import re
from bs4 import BeautifulSoup
from collections import Counter
import json
import os
import random
from rank_bm25 import BM25Okapi
import argparse
import subprocess
import argparse

class CommentsDatabase:

    '''the class creates a pool comments simply tokenized'''
    def __init__(self, json_content):
        self.candidates = []
        self.candidate_ids = []
        self.create_comments_pool(json_content)

    def tokenize(self, text):
        return text.lower().split()

    def create_comments_pool(self, json_content):

        for post_id in json_content:
            # print(post_id)
            post = json_content[post_id]
            post_type = post['post_type']
            post_body = post['body']

            if post_type == '2':
                self.candidates.append(self.tokenize(post_body))
                self.candidate_ids.append(post_id)
        
        print("creating comments pool ......")
        print("%d comments are loaded" %len(self.candidates))
        # print(self.candidates[0])
        self.bm25 = BM25Okapi(self.candidates)

    def retreival_ranked_list(self, query, n):
        '''return 
            top ids, text, scores
        '''

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        sorted_scores = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)

        # res = [(self.candidate_ids[sorted_scores[i][0]], \
        #         self.candidates[sorted_scores[i][0]], \
        #         sorted_scores[i][1]) for i in range(min(n, len(sorted_scores))]
        res_ids = [self.candidate_ids[sorted_scores[i][0]] for i in range(min(n, len(sorted_scores)))]
        res_texts = [self.candidates[sorted_scores[i][0]] for i in range(min(n, len(sorted_scores)))]
        res_scores = [sorted_scores[i][1] for i in range(min(n, len(sorted_scores)))]

        # print("retreived ids")
        # print(res_ids)

        return res_ids, res_texts, res_scores
                

    def get_all_comments_ids(self):
        return self.candidate_ids


class AskUbuntuData:

    def __init__(self):
        self.TYPE_QUESTION = '1'
        self.TYPE_ANSWER = '2'

    def text_normalization(self, text):
        soup = BeautifulSoup(text)
        text = soup.get_text()
        text = text.replace('\n', ' '). replace('\r', ' ')
        text = ' '.join(text.split())
        return text

    def tags_normalization(self,text):
        text = text.replace('<', ' ').replace('>', ' ').replace('-', ' ')
        text = ' '.join(text.split())
        return text

    def xml2json(self, xml_filename, json_obj=None, output_json_filename_content=None, output_json_filename_relation=None, save_results=True):

        '''Args
            json_obj: if it is None, a new json is created, it is not None. Continue adding contens to the json'''

        # outfilename = xml_filename.replace(".xml", ".tsv")

        assert '.xml' in xml_filename, 'need .xml file'

        tree = ET.parse(xml_filename)
        root = tree.getroot()

        content_dict = {} #each content has an entry (post or comment)
        relation_dict = defaultdict(list) # the key is post id. The list contains all comments
        type_counts = Counter()


        for child in root:
            post_type = child.get('PostTypeId')
            post_id = child.get('Id')
            assert post_id, 'post id is None'
            assert post_id not in content_dict, 'duplicate entry'
            type_counts.update(post_type)

            if post_type == '1':
                #a question
                title = child.get('Title')
                body = child.get('Body')
                acceptedAnswerId = child.get('AcceptedAnswerId') # can be None
                tags = child.get('Tags')
                ownerUserId = child.get('OwnerUserId')
                commentCount = child.get('CommentCount')
                answerCount = child.get('AnswerCount')
                favoriteCount = child.get('favoriteCount')
                assert title, 'title is none'
                assert body, 'body is none'
                assert tags, 'tags is None'
                # if acceptedAnswerId is None:
                #     acceptedAnswerId = '-1'
                content_dict[post_id] = {'post_id': post_id,
                                         'post_type': post_type,
                                         'title': title,
                                         'body': self.text_normalization(body),
                                         'acceptedAnswerId': acceptedAnswerId,
                                         'tags': self.tags_normalization(tags),
                                         'ownerUserId': ownerUserId,
                                         'commentCount': commentCount,
                                         'answerCount': answerCount,
                                         'favoriteCount': favoriteCount
                                         }

                # print(content_dict)
                # exit(1)

            elif post_type == '2':
                #an answer
                parentId = child.get('ParentId')
                body = child.get('Body')
                assert parentId, 'parentId is None'
                assert body, 'comment body is None'

                content_dict[post_id] = {'post_id': post_id,
                                         'post_type': post_type,
                                         'parentId': parentId,
                                         'body': self.text_normalization(body)
                                         }
                relation_dict[parentId].append(post_id)
            else:
                # print('unknown type {}   id {}'.format(post_type, post_id))
                '''we will ignore other types
                type 4: seems like post with subject
                type 5: seems deleted posts
                '''
                continue
        
        if save_results:
            #if we want to save the results, check or create output file name
            if output_json_filename_content is None:
                output_json_filename_content = xml_filename.replace('.xml', '_content.json')
            if output_json_filename_relation is None:
                output_json_filename_relation = xml_filename.replace('.xml', '_relation.json')

            with open(output_json_filename_content, 'w', encoding='utf-8') as fdata,\
                open(output_json_filename_relation, 'w', encoding='utf-8') as frelation:
                json.dump(content_dict, fdata)
                json.dump(relation_dict, frelation)

        print("type counts")
        print(type_counts)

        return content_dict, relation_dict

    def extract_all_questions(self, xml_filename=None, json_content_filename=None, out_filename=None):

        '''extract all questions from the forum and save to a file'''
        if json_content_filename:
            with open(json_content_filename, 'r', encoding='utf-8') as fin:
                json_content = json.load(fin)
        elif xml_filename:
            json_content, json_relation = self.xml2json(xml_filename)
        else:
            raise Exception('xml_file or json_obj needs to be given')
        
        it = 0
        with open(out_filename, 'w', encoding='utf-8') as fout:
            for post_id in json_content:
                if json_content[post_id]['post_type'] == self.TYPE_QUESTION:
                    it += 1
                    if it %1000 == 0:
                        print(it)
                    q = json_content[post_id]
                    # concat_q = '%s %s' %(q['title'], q['body'])
                    # post_id = q['post_id']
                    fout.write('%s\t%s\t%s\n' %(q['post_id'], q['body'], q['title']))
        return out_filename

    def refine_context(text):
        '''some context contains Possible Duplicate: ....
        Remove this information because it gives the label?
        '''
        return
                    
    def generate_drp_training_data(self, data_file,
                                   out_filename,
                                   debug_file,
                                   xml_filename=None,
                                   json_content_filename=None,
                                   # json_relation_filename=None,
                                   q1method='bs',
                                   q2method='bs'):
        '''
        DPR train/dev/testing data are json file has the following format
        [
            {
                "dataset": "nq_dev_psgs_w100",
                "question": "who sings does he love me with reba",
                "answers": [
                    "Linda Davis"
                ],
            "positive_ctxs": [
                {
                    "title": "Does He Love You",
                    "text": "Does He Love You \"Does He Love You\" is a song written by Sandy Knox and Billy Stritch, and recorded as a duet by American country music artists Reba McEntire and Linda Davis. It was released in August 1993 as the first single from Reba's album \"Greatest Hits Volume Two\". It is one of country music's several songs about a love triangle. \"Does He Love You\" was written in 1982 by Billy Stritch. He recorded it with a trio in which he performed at the time, because he wanted a song that could be sung by the other two members",
                    "score": 1000,
                    "title_score": 1,
                    "passage_id": "11828866"
                }
            ],
            "negative_ctxs": [
                {
                    "title": "Cormac McCarthy",
                    "text": "chores of the house, Lee was asked by Cormac to also get a day job so he could focus on his novel writing. Dismayed with the situation, she moved to Wyoming, where she filed for divorce and landed her first job teaching. Cormac McCarthy is fluent in Spanish and lived in Ibiza, Spain, in the 1960s and later settled in El Paso, Texas, where he lived for nearly 20 years. In an interview with Richard B. Woodward from \"The New York Times\", \"McCarthy doesn't drink anymore \u2013 he quit 16 years ago in El Paso, with one of his young",
                    "score": 0,
                    "title_score": 0,
                    "passage_id": "2145653"
                }
            ],
            "hard_negative_ctxs": [
                {
                    "title": "Why Don't You Love Me (Beyonce\u0301 song)",
                    "text": "song. According to the lyrics of \"Why Don't You Love Me\", Knowles impersonates a woman who questions her love interest about the reason for which he does not value her fabulousness, convincing him she's the best thing for him as she sings: \"Why don't you love me... when I make me so damn easy to love?... I got beauty... I got class... I got style and I got ass...\". The singer further tells her love interest that the decision not to choose her is \"entirely foolish\". Originally released as a pre-order bonus track on the deluxe edition of \"I Am...",
                    "score": 14.678405,
                    "title_score": 0,
                    "passage_id": "14525568"
                }
        ]
    }
]
        '''
        # assert xml_filename or json_obj, 'xml_file or json_obj needs to be given'

        # if json_content_filename and json_relation_filename:
        if json_content_filename:
            with open(json_content_filename, 'r', encoding='utf-8') as fin:
                json_content = json.load(fin)
            # with open(json_relation_filename, 'r', encoding='utf-8') as fin:
            #     json_relation = json.load(fin)

        elif xml_filename:
            json_content, json_relation = self.xml2json(xml_filename)
        else:
            raise Exception('xml_file or json_obj needs to be given')

        # print(len(json_content.keys()))
        # print(json_relation['211039'])
        # print(json_content['211039'])
        # print(json_content['262144'])
        # print(json_content['211051'])
        print(data_file)

        samples = []
        all_data = []
        with open(data_file, 'r', encoding='utf-8') as fin,\
            open(debug_file, 'w', encoding='utf-8') as fdebug:
            num_questions = 0
            for line in fin:
                # if len(line.strip().split('\t')) != 3:
                #     import pdb; pdb.set_trace()
                #     pass
                num_questions += 1
                one_example = {'dataset': 'askubuntu', 'answers': ['no answer']}
                base_qid, pos_qids, neg_qids = line.strip().split('\t')[:3]
                pos_qids = set(pos_qids.split(' '))
                neg_qids = set(neg_qids.split(' '))
                neg_qids = neg_qids.difference(pos_qids)

                orig_post = json_content[base_qid]
                # if len(pos_qids) == 0 or len(neg_qids) == 0:
                #     # print(line)
                #     # print(pos_qids)
                #     # print(neg_qids)
                #     print(line)
                
                #for question
                if q1method == 'b':
                    one_example['question'] = orig_post['body']
                elif q1method == 's':
                    one_example['question'] = orig_post['title']
                elif q1method == 'bs':
                    one_example['question'] = "%s %s" %(orig_post['title'], orig_post['body'])
                else:
                    raise NotImplementedError

                one_example['positive_ctxs'] = []

                for pos_qid in pos_qids:
                    if pos_qid == '':
                        continue
                    pos_post = json_content[pos_qid]

                    if q2method == 's':
                        one_example['positive_ctxs'].append({
                            "title": "",
                            "text": pos_post['title'],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": pos_post['post_id']
                        })
                    elif q2method == 'bs':
                        one_example['positive_ctxs'].append({
                            "title": pos_post['title'],
                            "text": pos_post['body'],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": pos_post['post_id']
                        })
                    else:
                        raise Exception("q2method value not found %s" %q2method)
                one_example['negative_ctxs'] = []
                one_example['hard_negative_ctxs'] = []
                for neg_qid in neg_qids:
                    if neg_qid == '':
                        continue
                    neg_post = json_content[neg_qid]

                    if q2method == 's':
                        one_example['hard_negative_ctxs'].append({
                            "title": "",
                            "text": neg_post['title'],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": neg_post['post_id']
                        })
                    elif q2method == 'bs':
                        one_example['hard_negative_ctxs'].append({
                            "title": neg_post['title'],
                            "text": neg_post['body'],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": neg_post['post_id']
                        })
                    else:
                        raise Exception("q2method value not found %s" %q2method)
                
                all_data.append(one_example)

                # if len(neg_qids) == 0:
                #     all_data.append(one_example)

                #     print(base_qid)
                #     all_data.append(one_example)
                #     fdebug.write(one_example['question'] + '\n\n')
                #     for p_p in one_example['positive_ctxs']:
                #         fdebug.write('%s\t%s\n' %(p_p['title'], p_p['text']))
                #     break
                
        with open(out_filename, 'w', encoding='utf-8') as fout:
            json.dump(all_data, fout)

        print("number of questions %d" %num_questions)


    def generate_question_based_comments_ranking_data(self, 
                                                    json_content_filename, 
                                                    json_relation_filename, 
                                                    outfilename_train,
                                                    outfilename_dev,
                                                    outfilename_test,
                                                    dev_ratio=0.05,
                                                    max_queries=10,
                                                    num_hard_negatives=10,
                                                    num_easy_negatives=10,
                                                    qmethod = 'bs',
                                                    num_shards=0,
                                                    shard_id=0
                                                    ):
        '''generate data for pairwise ranking task.
        for each question, generate a list of comments pairs.

        For one question, comments are in 4 groups. 
                            1) 0 or 1accepted comment (accepted_comment)
                            2) 0 to N associated comments   (associated_comments)
                            3) non-associated comments but have high text similarity scores (BM25?)   (hard_negative_comments)
                            4) the remaining comments (easy_negative comments)
       
        Sampling stratgegies:
        Positive: randomly sample accepted_comment and associated_comments. All positives will be added
        Hard negatives: top comments retrieved by BM25 but not positive
        Easy negatives: randomly sampled comments but are not in hard negatives and positive

        args:
            num_easy_negatives: number of easy negatives to be sampled
            num_hard_negatives
            max_queries: max number of queries in this function or this shard
            num_shards: number of shards. It <= 1, don't split into shards
        '''
        print("num_easy_negatives is %d    num_hard_negatives %d" %(num_easy_negatives, num_hard_negatives)) 
        assert num_shards >= 1, "number of shards needs to b >= 1"
        assert shard_id < num_shards

        if json_content_filename:
            with open(json_content_filename, 'r', encoding='utf-8') as fin:
                json_content = json.load(fin)
            with open(json_relation_filename, 'r', encoding='utf-8') as fin:
                json_relation = json.load(fin)

        comments_pool = CommentsDatabase(json_content)
        num_queries, num_train_queries, num_dev_queries, num_test_queries, number_of_ctx = 0,0,0,0,0
        train_data, dev_data, test_data = [], [], []
        total_num_of_queries = -1
        for post_id in json_content:
            post_type = json_content[post_id]['post_type']
            if post_type == '1' and post_id in json_relation:

                question = json_content[post_id]
                if not (question['acceptedAnswerId'] and question['acceptedAnswerId'] in json_content):
                    #no accepted Answer 
                    continue
                
                #mechanism to split data into multiple shards
                total_num_of_queries += 1
                if total_num_of_queries % num_shards != shard_id:
                    #use multiple machine to process data
                    continue

                if num_queries >= max_queries:
                    print("num of queries reached")
                    break
                num_queries += 1
                
                coin = random.random()
                if coin < dev_ratio:
                    all_data = test_data
                    num_test_queries += 1
                elif coin < (2 * dev_ratio):
                    all_data = dev_data
                    num_dev_queries += 1
                else:
                    all_data = train_data
                    num_train_queries += 1
                
                if num_queries%100 == 0:
                    print("%d queries processed. train: %d dev: %d test: %d"%(num_queries, num_train_queries, num_dev_queries, num_test_queries))
                    
                #this is a question
                #build 4 groups
                post_title = question['title']
                post_body = question['body']

                # print(post_id)
                # print(post_title)
                # print(post_body)
                accepted_comments_ids, associated_comments_ids, hard_negative_comments_ids, easy_negative_comments_ids = set(), set(), set(), set()
                
                accepted_comments_ids.add(question['acceptedAnswerId'])
                #accepted comments
                # if question['acceptedAnswerId'] and question['acceptedAnswerId'] in json_content:
                #     accepted_comment_ids.add(question['acceptedAnswerId'])

                #both accepted and associated comments
                for c_id in json_relation[post_id]:
                    if c_id in json_content:
                        associated_comments_ids.add(c_id)

                #hard negative ids
                if num_hard_negatives > 0:
                    #try to retrieve top N. Some may overlaps with positive
                    N = len(associated_comments_ids) + num_hard_negatives + 10
                    query = post_title + " " + post_body

                    re_ids, _, re_scores = comments_pool.retreival_ranked_list(query, N)

                    num_h = 0
                    for c_id in re_ids:
                        if c_id not in associated_comments_ids and c_id in json_content:
                            hard_negative_comments_ids.add(c_id)
                            num_h += 1
                            if num_h == num_hard_negatives:
                                break

                if num_easy_negatives > 0:
                    all_comments_ids = comments_pool.get_all_comments_ids()

                    #randomly sample irrelevant examples
                    num_e = 0
                    while num_e < num_easy_negatives:
                        rnum = random.randrange(0, len(all_comments_ids) - 1)
                        rid = all_comments_ids[rnum]
                        if rid in associated_comments_ids or rid in hard_negative_comments_ids or rid in easy_negative_comments_ids:
                            continue
                        else:
                            easy_negative_comments_ids.add(rid)
                            num_e += 1

                one_example = {'dataset': 'askubuntu', 'answers': [post_id]}
                if qmethod == 's':
                    one_example['question'] = post_title
                elif qmethod == 'bs':
                    one_example['question'] = "%s %s" %(post_title, post_body)
                elif qmethod == 'b':
                    one_example['question'] = post_body
                else:
                    raise NotImplementedError

                one_example['positive_ctxs'] = [
                    {
                        'title': "",
                        'text': json_content[c_id]['body'],
                        'score': 0,
                        'title_score': 0,
                        'passage_id': c_id
                    }
                    for c_id in accepted_comments_ids
                ]
                one_example['negative_ctxs'] = [
                    {
                        'title': "",
                        'text': json_content[c_id]['body'],
                        'score': 0,
                        'title_score': 0,
                        'passage_id': c_id  
                    }
                    for c_id in easy_negative_comments_ids
                ]
                one_example['hard_negative_ctxs'] = [
                    {
                        'title': "",
                        'text': json_content[c_id]['body'],
                        'score': 0,
                        'title_score': 0,
                        'passage_id': c_id
                    }
                    for c_id in hard_negative_comments_ids
                ]
                all_data.append(one_example)
                number_of_ctx = number_of_ctx + len(hard_negative_comments_ids) + len(easy_negative_comments_ids) + len(accepted_comments_ids)
        
        print("number of examples extracted. train: %d, dev: %d, test: %d" %(num_train_queries, num_dev_queries, num_test_queries))
        print("total number of queries %d" %total_num_of_queries)
        print("average number of context %f" %(float(number_of_ctx)/float(num_queries)))

        with open(outfilename_train, 'w', encoding='utf-8') as fout:
            json.dump(train_data, fout)
        
        with open(outfilename_dev, 'w', encoding='utf-8') as fout:
            json.dump(dev_data, fout)
        
        with open(outfilename_test, 'w', encoding='utf-8') as fout:
            json.dump(test_data, fout)

        return outfilename_train, outfilename_dev, outfilename_test

        


def main_generate_finetune_data():
    filename = '/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/askubuntu_2014_posts.xml'
    filename_content = '/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/askubuntu_2014_posts_content.json'
    filename_relation = '/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/askubuntu_2014_posts_relation.json'
    ad = AskUbuntuData()
    #the function to create json only needs to run once
    # content, relation = ad.xml2json(filename, output_json_filename_content=filename_content, output_json_filename_relation=filename_relation)
    filename_train = "/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/train_random.txt"
    filename_dev = "/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/dev.txt"
    filename_test = "/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/test.txt"
    base_path = '/private/home/xiaojianwu/projects/question_similarity/data/askubuntu'
    for t in ['c', 'r']: #classification or regression
        for i in ['b', 's', 'bs', 'bsc']:
            print("processing {}   {}".format(t, i))
            name = '{}_{}'.format(i, t)
            outfoldername = os.path.join(base_path, name)
            if not os.path.exists(outfoldername):
                os.mkdir(outfoldername)

            outfilename_train = os.path.join(outfoldername, 'train.tsv')
            outfilename_dev = os.path.join(outfoldername, 'dev.tsv')
            outfilename_test = os.path.join(outfoldername, 'test.tsv')
            
            ad.generate_finetune_data_from_xml(filename_train, outfilename_train, json_content_filename=filename_content, method=i)
            ad.generate_finetune_data_from_xml(filename_dev, outfilename_dev, json_content_filename=filename_content, method=i)
            ad.generate_finetune_data_from_xml(filename_test, outfilename_test, json_content_filename=filename_content, method=i)


def main_generate_dpr_qa_data():
    filename_content = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_content.json'
    filename_relation = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_relation.json'
    
    base_path = '/checkpoint/xiaojianwu/data/askubuntu/dpr'
    qmethod = 'bs'
    outfilename_train = os.path.join(base_path, 'askubuntu_qa_q%s_train.json' %qmethod)
    outfilename_dev = os.path.join(base_path, 'askubuntu_qa_q%s_dev.json' %qmethod)
    outfilename_test = os.path.join(base_path, 'askubuntu_qa_q%s_test.json' %qmethod)
    
    ad = AskUbuntuData()

    ad.generate_question_based_comments_ranking_data(filename_content, 
                                                    filename_relation, 
                                                    outfilename_train,
                                                    outfilename_dev,
                                                    outfilename_test,
                                                    dev_ratio=0.05,
                                                    max_queries=10,
                                                    num_hard_negatives=10,
                                                    num_easy_negatives=10,
                                                    qmethod = qmethod)

def combine_multiple_file():
    #combine multiple files
    num_shards = 30
    base_path = '/checkpoint/xiaojianwu/data/askubuntu/dpr/shards'
    for t in ['train', 'dev', 'test']:
        filenames = [os.path.join(base_path, 'askubuntu_qa_qbs_10_10_1000000_%s_shard_%d_%d.json' %(t, num_shards, shard_id)) for shard_id in range(num_shards)]
        outfilename = '/checkpoint/xiaojianwu/data/askubuntu/dpr/askubuntu_qa_qbs_10_10_1000000_%s_shard_%d_combined.json' %(t, num_shards)
        all_data = []
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fin:
                data = json.load(fin)
                all_data.extend(data)
        with open(outfilename, 'w', encoding='utf-8') as fout:
            json.dump(all_data, fout)
        print("number of examples %d" %len(all_data))

def rewrite_negatives():
    #to adapt to the current setting
    #make negatives empty and make hard negatives to have 1 hard negatives and a list of easy negatives
    for t in ['train', 'dev', 'test']:
        filename = '/checkpoint/xiaojianwu/data/askubuntu/dpr/askubuntu_qa_qbs_10_10_1000000_%s_shard_30_combined.json' %t
        outfilename = '/checkpoint/xiaojianwu/data/askubuntu/dpr/askubuntu_qa_qbs_10_10_1000000_%s_shard_30_combined_modified.json' %t
        num_true_hard_negative = 1
        with open(filename, 'r', encoding='utf-8') as fin:
            all_data = json.load(fin)
        for example in all_data:
            new_hard_negatives = [example['hard_negative_ctxs'][i] for i in range(num_true_hard_negative)] + example['negative_ctxs']
            example['negative_ctxs'] =[]
            example['hard_negative_ctxs'] = new_hard_negatives
        
        with open(outfilename, 'w', encoding='utf-8') as fout:
            json.dump(all_data, fout)

        #test
        print(all_data[0]['negative_ctxs'])
        print(len(all_data[0]['hard_negative_ctxs']))

        print("number of examples %d" %len(all_data))

def get_stats():
    filename = '/checkpoint/xiaojianwu/data/askubuntu/dpr/askubuntu_q1bs_q2bs_train.json'
    with open(filename, 'r', encoding='utf-8') as fin:
        all_data = json.load(fin)
    print(len(all_data))


def main_generate_dpr_qa_data_cmd():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_hard_negatives', type=int, default=10)
    parser.add_argument('--num_easy_negatives', type=int, default=10)
    parser.add_argument('--max_num_queries', type=int, default=10)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=0)
    parser.add_argument('--qmethod', type=str, default='bs')
    parser.add_argument('--output_folder', type=str, default='/checkpoint/xiaojianwu/data/askubuntu/dpr')

    args = parser.parse_args()

    filename_content = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_content.json'
    filename_relation = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_relation.json'
    
    base_path = args.output_folder
    qmethod = args.qmethod
    max_num_queries = args.max_num_queries
    num_hard_negatives = args.num_hard_negatives
    num_easy_negatives = args.num_easy_negatives
    num_shards = args.num_shards
    shard_id = args.shard_id

    outfilename_train = os.path.join(base_path, 'askubuntu_qa_q%s_%d_%d_%d_train_shard_%d_%d.json' %(qmethod, num_easy_negatives, num_hard_negatives, max_num_queries, num_shards, shard_id))
    outfilename_dev = os.path.join(base_path, 'askubuntu_qa_q%s_%d_%d_%d_dev_shard_%d_%d.json' %(qmethod, num_easy_negatives, num_hard_negatives, max_num_queries, num_shards, shard_id))
    outfilename_test = os.path.join(base_path, 'askubuntu_qa_q%s_%d_%d_%d_test_shard_%d_%d.json' %(qmethod, num_easy_negatives, num_hard_negatives, max_num_queries, num_shards, shard_id))
    
    ad = AskUbuntuData()
    ad.generate_question_based_comments_ranking_data(filename_content, 
                                                    filename_relation, 
                                                    outfilename_train,
                                                    outfilename_dev,
                                                    outfilename_test,
                                                    dev_ratio=0.05,
                                                    max_queries=max_num_queries,
                                                    num_hard_negatives=num_hard_negatives,
                                                    num_easy_negatives=num_easy_negatives,
                                                    qmethod = qmethod,
                                                    num_shards=num_shards,
                                                    shard_id=shard_id)


def main_generate_dpr_data():
    filename = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts.xml'
    all_questions = '/checkpoint/xiaojianwu/data/askubuntu/dpr/all_questions.tsv'
    json_content_filename = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_content.json'
    filename_content = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_content.json'
    filename_relation = '/checkpoint/xiaojianwu/data/askubuntu/orig/askubuntu_2014_posts_relation.json'
    ad = AskUbuntuData()
    # ad.extract_all_questions(xml_filename=filename, 
    #                         json_content_filename=json_content_filename, 
    #                         out_filename=all_questions)


    #generate training, dev, testing data
    filename_train = "/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/train_random.txt"
    filename_dev = "/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/dev.txt"
    filename_test = "/private/home/xiaojianwu/projects/question_similarity/data/askubuntu/test.txt"
    base_path = '/checkpoint/xiaojianwu/data/askubuntu/dpr'

    debug_file = '/checkpoint/xiaojianwu/data/askubuntu/dpr/debug.txt'

    # for q1m, q2m in [('s', 's'), ('s', 'bs'), ('bs', 'bs')]:
    for q1m, q2m in [('s', 'bs')]:

        print("processing {} {}".format(q1m, q2m))

        outfilename_train = os.path.join(base_path, 'askubuntu_q1%s_q2%s_train.json' %(q1m, q2m))
        outfilename_dev = os.path.join(base_path, 'askubuntu_q1%s_q2%s_dev.json' %(q1m, q2m))
        outfilename_test = os.path.join(base_path, 'askubuntu_q1%s_q2%s_test.json' %(q1m, q2m))

        test_file = os.path.join(base_path, 'test_q1%s_q2%s.json' %(q1m, q2m))
        
        ad.generate_drp_training_data(filename_train, outfilename_train, debug_file, json_content_filename=filename_content, q1method=q1m, q2method=q2m)
        ad.generate_drp_training_data(filename_dev, outfilename_dev, debug_file, json_content_filename=filename_content, q1method=q1m, q2method=q2m)
        ad.generate_drp_training_data(filename_test, outfilename_test, debug_file, json_content_filename=filename_content, q1method=q1m, q2method=q2m)


        # ad.generate_drp_training_data(filename_test, test_file, debug_file, json_content_filename=filename_content, q1method=q1m, q2method=q2m)




if __name__ == '__main__':


    
    # main_generate_dpr_data()
    # main_generate_dpr_qa_data()
    # main_generate_dpr_qa_data_cmd()
    # combine_multiple_file()
    get_stats()