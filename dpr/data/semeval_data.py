import xml.etree.ElementTree as ET
# import data_processing.post_cleaning_functions as pcf
# import data_processing.data_process_pipeline as dpp
# import data_processing.post_filtering_functions as pff
from collections import defaultdict
import json
import math
import os
from index import BM25Index
import random

#this version is different from my local version
class SemEvalData:

    def xml2json(self, filename, file2save_json=None):
        '''output json format
        The dict is indexed by original question id
        {"original question id": {subj: "", body: "", id: "", cand_qs: [{sub: "", body: "", id: "", label: "", comments[{text: "", label2orig: "", label2current: "", id: ""}] } ] } }
        '''
        res = {}
        outfilename = filename.replace(".xml", ".tsv")

        if '.xml' not in filename:
            raise Exception("need .xml file")
        tree = ET.parse(filename)
        root = tree.getroot()

        for child in root:

            orig_dict = {}
            cand_dict = {}
            orig_dict["id"] = child.get("ORGQ_ID")
            comments = []
            #get body
            for origQ in child.iter("OrgQBody"):
                orig_dict["body"] = origQ.text
            
            #get subject
            for origQ in child.iter("OrgQSubject"):
                orig_dict["sub"] = origQ.text

            #get candidate question
            for relQlabel in child.iter("RelQuestion"):
                cand_dict["label"] = relQlabel.get("RELQ_RELEVANCE2ORGQ")
                cand_dict["id"] = relQlabel.get("RELQ_ID")

            for relQ in child.iter("RelQBody"):
                cand_dict["body"] = relQ.text

            for relSub in child.iter("RelQSubject"):
                cand_dict["sub"] = relSub.text

            if cand_dict["body"] is None:
                cand_dict["body"] = ""
            if cand_dict["sub"] is None:
                cand_dict["sub"] = ""
            if orig_dict["body"] is None:
                orig_dict["body"] = ""
            if orig_dict["sub"] is None:
                orig_dict["sub"] = ""

            # print(len(child.find("Thread").findall("RelComment")))
            for i in child.find("Thread").findall("RelComment"):
                comment = {}
                comment["id"] = i.get("RELC_ID")
                comment["label2orig"] = i.get("RELC_RELEVANCE2ORGQ")
                comment["label2current"] = i.get("RELC_RELEVANCE2RELQ")
                # print(comment)

                for j in i.iter("RelCText"):
                    comment["text"] = j.text
                comments.append(comment)
            
            cand_dict["comments"] = comments

            

            if orig_dict["id"] not in res:
                orig_dict["cand_qs"] = [cand_dict]
                res[orig_dict["id"]] = orig_dict
                # print(orig_dict)
                # exit(1)
            else:
                res[orig_dict["id"]]["cand_qs"].append(cand_dict)

        if file2save_json:
            with open(file2save_json, 'w', encoding='utf-8') as fout:
                json.dump(res,fout)

        # print(res)
        return res

        # with open(outfilename, 'w', encoding="utf-8") as fout:
        #     for d in data:
        #         if d["label"] == "PerfectMatch":
        #             num_label = 2
        #         elif d["label"] == "Relevant":
        #             num_label = 1
        #         elif d["label"] == "Irrelevant":
        #             num_label = 0

        #         fout.write("%s\t%s\t%s\t%s\t%d\n" %(d["q1_id"], d["q2_id"], d["q1"], d["q2"], num_label))
        # return outfilename


    def extract_qqs(self, filename, outfilename, q1method='bs', q2method='bs'):
        '''
        filename: xml
        outfilename: should be train.tsv, dev.tsv, test.tsv
        type:
        b: body_only
        bs: body_subject
            both questions output body and subject 
                        orig_q_sub  orig_q_body  cand_q_sub  cand_q_body  qqs_label
        bsc: body_subject_one_comment
            I will randomly select a 'good' comment but not Bad or PotentialUseful. if no good comments exist, I will not select any comments.
            comments will be only selected for candidate questions
            output tsv file will have   
                        orig_q_sub  orig_q_body  cand_q_sub  cand_q_body  cand_q_comment  qqs_label
        '''
        json_data = self.xml2json(filename)
        all_data = []
        number_of_ctx = 0
        for orig_q_id in json_data:
            orig_q = json_data[orig_q_id]

            one_example = {'dataset': 'semeval', 'answers': [orig_q_id]}
            if q1method == 's':
                one_example['question'] = orig_q['sub']
            elif q1method == 'bs':
                one_example['question'] = "%s %s" %(orig_q['sub'], orig_q['body'])
            elif q1method == 'b':
                one_example['question'] = orig_q['body']
            else:
                raise NotImplementedError

            one_example['positive_ctxs'] = []
            one_example['negative_ctxs'] = []
            one_example['hard_negative_ctxs'] = []
            
            for cand_q in orig_q['cand_qs']:
                # first_good_comment = ''
                # for comment in cand_q['comments']:
                #     if comment['label2current'] == 'Good':
                #         first_good_comment = comment['text']
                #         break
                ctxs_type = 'positive_ctxs' if self.label2binary(cand_q['label']) == 1 else 'hard_negative_ctxs'
                if q2method == 's':
                    one_example[ctxs_type].append({
                        "title": "",
                        "text": cand_q['body'],
                        "score": 0,
                        "title_score": 0,
                        "passage_id": cand_q['id']
                    })
                elif q2method == 'bs':
                    one_example[ctxs_type].append({
                        "title": cand_q['sub'],
                        "text": cand_q['body'],
                        "score": 0,
                        "title_score": 0,
                        "passage_id": cand_q['id']
                    })
                elif q2method == 'b':
                    one_example[ctxs_type].append({
                        "title": "",
                        "text": cand_q['body'],
                        "score": 0,
                        "title_score": 0,
                        "passage_id": cand_q['id']
                    })                        
                else:
                    raise Exception("q2method value not found %s" %q2method)
                    
            all_data.append(one_example)
            number_of_ctx += (len(one_example['positive_ctxs']) + len(one_example['hard_negative_ctxs']))
        
        print("number of examples extracted %d" %len(all_data))
        print("average number of context %f" %(float(number_of_ctx)/float(len(all_data))))

        with open(outfilename, 'w', encoding='utf-8') as fout:
            json.dump(all_data, fout)
        
    def label2binary(self, label):
        qqs_label2binary = {"Irrelevant": 0, "Relevant": 1, "PerfectMatch": 1}
        return qqs_label2binary[label] 

    def qqs_label_numerize(self, label):
        qqs_label2num = {"Irrelevant": 0, "Relevant": 1, "PerfectMatch": 2}
        return qqs_label2num[label]


    def generate_question_answer_data(self, filenames, outfilename, q1method='bs'):
        '''
        for each question, generate positive and negative data
        '''
        number_of_ctx = 0
        all_data = []
        for filename in filenames:
            json_data = self.xml2json(filename)
            
            for orig_q_id in json_data:
                orig_q = json_data[orig_q_id]

                for cand_q in orig_q['cand_qs']:

                    one_example = {'dataset': 'semeval', 'answers': [cand_q['id']]}
                    one_example['positive_ctxs'] = []
                    one_example['negative_ctxs'] = []
                    one_example['hard_negative_ctxs'] = []

                    if q1method == 's':
                        one_example['question'] = cand_q['sub']
                    elif q1method == 'bs':
                        one_example['question'] = "%s %s" %(cand_q['sub'], cand_q['body'])
                    elif q1method == 'b':
                        one_example['question'] = cand_q['body']
                    else:
                        raise NotImplementedError

                    for comment in cand_q['comments']:
                        ctxs_type = 'positive_ctxs' if comment['label2current'] == 'Good' else 'hard_negative_ctxs'
                        
                        one_example[ctxs_type].append({
                        "title": "",
                        "text": comment['text'],
                        "score": 0,
                        "title_score": 0,
                        "passage_id": comment['id']
                        })

                    all_data.append(one_example)
                    number_of_ctx += (len(one_example['positive_ctxs']) + len(one_example['hard_negative_ctxs']))

        print("number of examples extracted %d" %len(all_data))
        print("average number of context %f" %(float(number_of_ctx)/float(len(all_data))))

        with open(outfilename, 'w', encoding='utf-8') as fout:
            json.dump(all_data, fout)

    
    def generate_answer_simiarity_data(self, filenames, out_filename, num_retrieved_negatives=1, num_random_negatives=5):

        #load json file
        json_datas = []
        for filename in filenames:
            json_datas.append(self.xml2json(filename))
        
        #get all comments
        all_comments, comment_ids = [], []
        for json_data in json_datas:
            for orig_q_id in json_data:
                orig_q = json_data[orig_q_id]
                for cand_q in orig_q['cand_qs']:
                    for comment in cand_q['comments']:
                        all_comments.append(comment['text'])
                        comment_ids.append(comment['id'])
        
        bm25_index = BM25Index(all_comments, comment_ids)
        #generate data
        #use the first positive comment as question and all remaining positive comments as positive
        #use negative comments as hard negative and also use BM25 to retrieve several hard negatives
        all_data = []
        number_of_ctx = 0
        num_queries = 0
        for json_data in json_datas:
            for orig_q_id in json_data:
                # if num_queries > 10:
                #     break
                orig_q = json_data[orig_q_id]
                for cand_q in orig_q['cand_qs']:
                    
                    if num_queries%100 == 0:
                        print("%d queries processed. "%(num_queries))
                    num_queries += 1
                    positive_comments, negative_comments = [], []
                    for comment in cand_q['comments']:
                        if comment['label2current'] == 'Good':
                            positive_comments.append(comment)
                        else:
                            negative_comments.append(comment)
                    
                    if len(positive_comments) > 1:
                        #needs to have two positive. One is used as question. One is used as positive.
                        #otherwise, skip this thread
                        #first positive comment as question
                        one_example = {'dataset': 'semeval_aas', 'answers': [positive_comments[0]['id']], 'question': positive_comments[0]['text']}
                        one_example['negative_ctxs'] = [] #not using at this point

                        #add positive
                        one_example['positive_ctxs'] = [
                            {
                            "title": "",
                            "text": positive_comments[i]['text'],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": positive_comments[i]['id']
                            }
                        for i in range(1, len(positive_comments))
                        ]
                        one_example['hard_negative_ctxs'] = [
                            {
                            "title": "",
                            "text": c['text'],
                            "score": 0,
                            "title_score": 0,
                            "passage_id": c['id']
                         }
                        for c in negative_comments
                        ]

                        #retrieve more hard negatives
                        if num_retrieved_negatives > 0:
                            query = positive_comments[0]['text']
                            N = len(negative_comments) + len(positive_comments) + num_retrieved_negatives
                            negative_ids = [c['id'] for c in negative_comments]
                            positive_ids = [c['id'] for c in positive_comments]

                            #randomly sample more hard negatives
                            re_ids, re_texts, re_scores = bm25_index.retreival_ranked_list(query, N)

                            num_r = 0
                            for j in range(len(re_ids)):
                                if re_ids[j] not in negative_ids and re_ids[j] not in positive_ids:
                                    one_example['hard_negative_ctxs'].append({
                                        "title": "",
                                        "text": re_texts[j],
                                        "score": 0,
                                        "title_score": 0,
                                        "passage_id": re_ids[j]
                                    })
                                    negative_ids.append(re_ids[j])
                                    num_r += 1
                                    if num_r >= num_retrieved_negatives:
                                        break
                        
                        if num_random_negatives > 0:
                            all_comments_ids = bm25_index.get_all_ids()
                            all_orig_comments = bm25_index.get_all_orig_texts()

                            #randomly sample irrelevant examples
                            num_e = 0
                            while num_e < num_random_negatives:
                                rnum = random.randrange(0, len(all_comments_ids) - 1)
                                rid = all_comments_ids[rnum]
                                if rid in negative_ids or rid in positive_ids:
                                    continue
                                else:
                                    one_example['hard_negative_ctxs'].append({
                                        "title": "",
                                        "text": all_orig_comments[rnum],
                                        "score": 0,
                                        "title_score": 0,
                                        "passage_id": all_comments_ids[rnum]
                                    })
                                    negative_ids.append(all_comments_ids[rnum])
                                    num_e += 1

                all_data.append(one_example)
                number_of_ctx += (len(one_example['positive_ctxs']) + len(one_example['hard_negative_ctxs']))

        print("number of examples extracted %d" %len(all_data))
        print("average number of context %f" %(float(number_of_ctx)/float(len(all_data))))

        with open(out_filename, 'w', encoding='utf-8') as fout:
            json.dump(all_data, fout)


    # def generate_ranking_data_for_taskB(self, filename, outfilename, type='b'):
    #     '''
    #     for each orig question, if there are M candidate questions. Potentially, we will create M*2 data point.
    #     We will not randomly swap two examples because this is a two-tower pairwise ranking objetive and pointwise model is shared.
    #     '''

    #     json_data = self.xml2json(filename)

    #     with open(outfilename, 'w', encoding='utf-8') as fout:
    #         for orig_q_id in json_data:
    #             orig_q = json_data[orig_q_id]

    #             cand_list = []

    #             for cand_q in orig_q['cand_qs']:
    #                 first_good_comment = ''
    #                 for comment in cand_q['comments']:
    #                     if comment['label2current'] == 'Good':
    #                         first_good_comment = comment['text']
    #                         break
    #                 #[sub, body, comment, label]
    #                 cand_list.append((cand_q['sub'], cand_q['body'], first_good_comment, self.qqs_label_numerize(cand_q['label'])))

    #             #create data by pairwise every two examples with different labels
    #             for i in range(len(cand_list)):
    #                 for j in range(i + 1, len(cand_list)):
    #                     if cand_list[i][-1] != cand_list[j][-1]:
    #                         candi = cand_list[i]
    #                         candj = cand_list[j]
    #                         if candi[-1] > candj[-1]:
    #                             label = '0'
    #                         else:
    #                             label = '1'

    #                         if type =='b':
    #                             out_s = '\t'.join([orig_q['body'], candi[1], candj[1], label])

    #                         if type =='bs':
    #                             out_s = '\t'.join([orig_q['sub'], orig_q['body'], candi[0], candi[1], candj[0], candj[1], label])

    #                         if type == 'bsc':
    #                             out_s = '\t'.join([orig_q['sub'], orig_q['body'], candi[0], candi[1], candi[2], candj[0], candj[1], candj[2], label])
                        
    #                         fout.write(out_s + '\n')

    # def generate_question_answer_data(self, xml_filename, outfilename):
    #     '''the goal is for pretrain
    #         the original file is very large, so we directly parse it and output results
    #     '''
        
    #     '''output json format
    #     The dict is indexed by original question id
    #     {"original question id": {subj: "", body: "", id: "", cand_qs: [{sub: "", body: "", id: "", label: "", comments[{text: "", label2orig: "", label2current: "", id: ""}] } ] } }
    #     '''
    #     # res = {}
    #     # outfilename = xml_filename.replace('.xml', '_qa.tsv')

    #     if '.xml' not in xml_filename:
    #         raise Exception("need .xml file")
    #     tree = ET.parse(xml_filename)
    #     root = tree.getroot()
    #     assert '.tsv' in outfilename
    #     i = 0
    #     with open(outfilename, 'w', encoding='utf-8') as fout:
    #         for child in root:

    #             if i %10000 == 0:
    #                 print(i)
    #             i += 1

    #             # answer, relQ_body, relQ_sub = "", "", ""
    #             for relQ in child.iter("RelQBody"):
    #                 relQ_body = relQ.text

    #             for relSub in child.iter("RelQSubject"):
    #                 relQ_sub = relSub.text

    #             for relAnswer in child.iter("RelAnswer"):
    #                 accepted = relAnswer.get("RELA_ACCEPTED")
    #                 for RelAText in relAnswer.iter("RelAText"):
    #                     answer = RelAText.text ## here

    #                     # print(answer)

    #                     if relQ_sub is None:
    #                         relQ_sub = ""
    #                     if relQ_body is None:
    #                         relQ_body = ""
    #                     if answer is None:
    #                         answer = ""
    #                     if accepted is None:
    #                         accepted = "0"

    #                     fout.write("%s\t%s\t%s\t%s\n" %(relQ_sub, relQ_body, answer, accepted))
    #             # exit(1)
    #             # print(relQ_body, relQ_sub, answer, accepted)    
    #             # print("")
    #             # exit(1)    

    #     # print(res)
    #     return res



# def main_generate_classification_regression_taskB():
#     sed = SemEvalData()
#     # filename_train1 = "/private/home/xiaojianwu/projects/question_similarity/data/SemEval/SemEval2016-Task3-CQA-QL-train-part1.xml"
#     # filename_train2 = "/private/home/xiaojianwu/projects/question_similarity/data/SemEval/SemEval2016-Task3-CQA-QL-train-part2.xml"
#     # filename_Dev = "/private/home/xiaojianwu/projects/question_similarity/data/SemEval/SemEval2016-Task3-CQA-QL-dev.xml"
#     # filename_test17 = "/private/home/xiaojianwu/projects/question_similarity/data/SemEval/SemEval2017-task3-English-test.xml"
#     test = "/private/home/xiaojianwu/projects/question_similarity/data/SemEval/test.xml"
#     # sed.extract_taskB_data(filename_train1)
#     # sed.extract_taskB_data(filename_train2)
#     # sed.extract_taskB_data(filename_Dev)
#     filename_train1 = "/private/home/xiaojianwu/projects/question_similarity/data/semeval/SemEval2016-Task3-CQA-QL-train-part1.xml"
#     filename_train_combine = "/private/home/xiaojianwu/projects/question_similarity/data/semeval/SemEval2016-Task3-CQA-QL-train-part1_combine_1_2_dev.xml"
#     filename_Dev = "/private/home/xiaojianwu/projects/question_similarity/data/semeval/SemEval2016-Task3-CQA-QL-test.xml"
#     filename_test17 = "/private/home/xiaojianwu/projects/question_similarity/data/semeval/SemEval2017-task3-English-test.xml"
#     base_path = '/private/home/xiaojianwu/projects/question_similarity/data/semeval'
#     for t in ['c', 'r']: #classification or regression
#         for i in ['b', 'bs', 'bsc']:
#             name = '{}_{}'.format(i, t)
#             outfoldername = os.path.join(base_path, name)
#             if not os.path.exists(outfoldername):
#                 os.mkdir(outfoldername)

#             outfilename_train = os.path.join(outfoldername, 'train.tsv')
#             outfilename_dev = os.path.join(outfoldername, 'dev.tsv')
#             outfilename_test = os.path.join(outfoldername, 'test.tsv')

#             sed.extract_taskB(filename_train1, outfilename_train, type=i)
#             sed.extract_taskB(filename_Dev, outfilename_dev, type=i)
#             sed.extract_taskB(filename_test17, outfilename_test, type=i)

def main_generate_qa_dpr_data():
    sed = SemEvalData()
    # ad.extract_all_questions(xml_filename=filename, 
    #                         json_content_filename=json_content_filename, 
    #                         out_filename=all_questions)
    #generate training, dev, testing data
    filename_train1 = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-train-part1.xml"
    filename_train_combine = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-train-part1_combine_1_2_dev.xml"
    filename_dev = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-test.xml"
    filename_test17 = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2017-task3-English-test.xml"
    base_path = '/checkpoint/xiaojianwu/data/semeval/dpr'

    # debug_file = '/checkpoint/xiaojianwu/data/semeval/dpr/debug.txt'

    # for q1m, q2m in [('s', 's'), ('s', 'bs'), ('bs', 'bs')]:
    for q1m in ['bs']:

        print("processing {}".format(q1m))

        outfilename_train = os.path.join(base_path, 'semeval_qa_q1%s_train.json' %q1m)
        outfilename_dev = os.path.join(base_path, 'semeval_qa_q1%s_dev.json' %q1m)
        outfilename_test = os.path.join(base_path, 'semeval_qa_q1%s_test.json' %q1m)
        outfilename_merged = os.path.join(base_path, 'semeval_qa_q1%s_merged.json' %q1m)
        outfilename_all = os.path.join(base_path, 'semeval_qa_q1%s_all.json' %q1m)
        # test_file = os.path.join(base_path, 'test_q1%s_q2%s.json' %(q1m, q2m))
        sed.generate_question_answer_data([filename_train1], outfilename_train, q1method=q1m)
        sed.generate_question_answer_data([filename_dev], outfilename_dev, q1method=q1m)
        sed.generate_question_answer_data([filename_test17], outfilename_test, q1method=q1m)

        sed.generate_question_answer_data([filename_train_combine, filename_dev], outfilename_merged, q1method=q1m)
        sed.generate_question_answer_data([filename_train_combine, filename_dev, filename_test17], outfilename_all, q1method=q1m)

def main_generate_aas_dpr_data():
    #generate answer similarity data
    sed = SemEvalData()
    # ad.extract_all_questions(xml_filename=filename, 
    #                         json_content_filename=json_content_filename, 
    #                         out_filename=all_questions)
    #generate training, dev, testing data
    filename_train1 = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-train-part1.xml"
    filename_train_combine = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-train-part1_combine_1_2_dev.xml"
    filename_dev = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-test.xml"
    filename_test17 = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2017-task3-English-test.xml"
    base_path = '/checkpoint/xiaojianwu/data/semeval/dpr'

    # debug_file = '/checkpoint/xiaojianwu/data/semeval/dpr/debug.txt'
    # for q1m, q2m in [('s', 's'), ('s', 'bs'), ('bs', 'bs')]:
    num_retrieved_neg = 2
    num_random_neg = 10
    outfilename_train = os.path.join(base_path, 'semeval_aas_%d_train.json' %num_retrieved_neg)
    outfilename_dev = os.path.join(base_path, 'semeval_aas_%d_dev.json' %num_retrieved_neg)
    outfilename_test = os.path.join(base_path, 'semeval_aas_%d_test.json' %num_retrieved_neg)
    outfilename_merged = os.path.join(base_path, 'semeval_aas_%d_merged.json' %num_retrieved_neg)
    outfilename_all = os.path.join(base_path, 'semeval_aas_%d_all.json' %num_retrieved_neg)
    # test_file = os.path.join(base_path, 'test_q1%s_q2%s.json' %(q1m, q2m))
    sed.generate_answer_simiarity_data([filename_train1], outfilename_train, num_retrieved_neg, num_random_neg)
    # sed.generate_answer_simiarity_data([filename_dev], outfilename_dev, num_retrieved_neg, num_random_neg)
    # sed.generate_answer_simiarity_data([filename_test17], outfilename_test, num_retrieved_neg, num_random_neg)
    # sed.generate_answer_simiarity_data([filename_train_combine, filename_dev], outfilename_merged, num_retrieved_neg, num_random_neg)
    # sed.generate_answer_simiarity_data([filename_train_combine, filename_dev, filename_test17], outfilename_all, num_retrieved_neg, num_random_neg)


def main_generate_qq_dpr_data():
    all_questions = '/checkpoint/xiaojianwu/data/semeval/dpr/all_questions.tsv'
    sed = SemEvalData()
    # ad.extract_all_questions(xml_filename=filename, 
    #                         json_content_filename=json_content_filename, 
    #                         out_filename=all_questions)
    #generate training, dev, testing data
    filename_train1 = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-train-part1.xml"
    filename_train_combine = "/checkpoint/xiaojianwu/data/semeval/orig/semeval/SemEval2016-Task3-CQA-QL-train-part1_combine_1_2_dev.xml"
    filename_dev = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2016-Task3-CQA-QL-test.xml"
    filename_test17 = "/checkpoint/xiaojianwu/data/semeval/orig/SemEval2017-task3-English-test.xml"
    base_path = '/checkpoint/xiaojianwu/data/semeval/dpr'

    # debug_file = '/checkpoint/xiaojianwu/data/semeval/dpr/debug.txt'

    # for q1m, q2m in [('s', 's'), ('s', 'bs'), ('bs', 'bs')]:
    for q1m, q2m in [('s', 's'), ('bs', 'bs'), ('s', 'bs')]:

        print("processing {} {}".format(q1m, q2m))

        outfilename_train = os.path.join(base_path, 'semeval_q1%s_q2%s_train.json' %(q1m, q2m))
        outfilename_dev = os.path.join(base_path, 'semeval_q1%s_q2%s_dev.json' %(q1m, q2m))
        outfilename_test = os.path.join(base_path, 'semeval_q1%s_q2%s_test.json' %(q1m, q2m))

        # test_file = os.path.join(base_path, 'test_q1%s_q2%s.json' %(q1m, q2m))
        sed.extract_qqs(filename_train1, outfilename_train, q1method=q1m, q2method=q2m)
        sed.extract_qqs(filename_dev, outfilename_dev, q1method=q1m, q2method=q2m)
        sed.extract_qqs(filename_test17, outfilename_test, q1method=q1m, q2method=q2m)

if __name__ == "__main__":


    # main_generate_qq_dpr_data()
    # main_generate_qa_dpr_data()
    main_generate_aas_dpr_data()