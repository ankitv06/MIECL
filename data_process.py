import os
from re import S
from torch._C import dtype
from nltk.tokenize import word_tokenize
import numpy as np
import random
import torch
import json

#import nltk
#nltk.download('punkt')

class DataProcess():
    def __init__(self, file1, file2, file3, file4, file5, file6):
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        self.file5 = file5
        self.file6 = file6

        self.news_id = {'NULL': 0}
        self.title_content = {}
        self.abstract_content = {}
        self.word_dict = {'PADDING': 0}
        self.entity_dict = {'PADDING': 0}
        self.news_title_dict = {'0': [0] * 20}
        self.news_abstract_dict = {'0': [0] * 40}
        self.news_entity_dict = {'0': [0] * 5}
        self.newsid_topic = {0: 'NULL'}
        self.news_category    = {}  # nid -> category string
        self.news_subcategory = {}  # nid -> subcategory string
        self.entity_news = {}
        self.entity_matrix_dict = {0: np.zeros(100, dtype='float32')}
        self.embedding_dict = {}

        self.userid_dict = {'NULL': 0}
        self.npratio1 = 4
        self.npratio2 = 50
        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.train_user = []
        # popularity debiasing triplet lists
        self.train_pop   = []
        self.train_unpop = []
        self.train_diff  = []
        self.user_his_pad = {0: [0] * 50, }
        self.user_his_complete = {0: [], }
        # echo-chamber debiasing
        self.user_his_echo_pad = {0: [0] * 50, }
        # popularity pools
        self.popular_set      = set()
        self.nonpopular_set   = set()
        self.popular_pool     = {}   # topic_key -> [popular nids]
        self.nonpopular_pool  = {}   # topic_key -> [non-popular nids]
        self.category_pool    = {}   # category -> [all nids in category]
        self._get_topic_key   = None
        self._topic_to_all    = {}
        self._all_topic_keys  = []

        self.val_index = []
        self.val_candidate = []
        self.val_label = []
        self.val_user_his = []
        self.val_user = []

        self.test_index = []
        self.test_candidate = []
        self.test_label = []
        self.test_user_his = []
        self.test_user = []

    # random sample of a specified size is obtained from the array
    # if npratio is lesser than len(array), samples directly from array
    # else creates a repeated version of the array for sampling
    def newsample(self, array, npratio):
        if npratio > len(array):
            return random.sample(array*(npratio // len(array) + 1), npratio)
        else:
            return random.sample(array, npratio)


    # 处理新闻数据

    # read and process news from a file containing news data
    def process_news(self, file):
        f = open(file, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            self.title_content[line[0]] = word_tokenize((line[3]).lower())
            self.abstract_content[line[0]] = word_tokenize((line[4]).lower())
            if line[0] not in self.news_id:
                self.news_id[line[0]] = len(self.news_id)
            nid = self.news_id[line[0]]
            # store category and subcategory for popularity pools
            self.news_category[nid]    = line[1] if len(line) > 1 else 'unknown'
            self.news_subcategory[nid] = line[2] if len(line) > 2 else 'unknown'
            self.newsid_topic[nid]     = self.news_category[nid]

            # iterate through the words of the title
            # assign unique ids to every word
            # limit length to 20
            # create a list title with each word represented by its unique ID
            title = []
            for word in self.title_content[line[0]]:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                title.append(self.word_dict[word])
            title = title[:20]
            if self.news_id[line[0]] not in self.news_title_dict:
                self.news_title_dict[self.news_id[line[0]]] = title + [0] * (20 - len(title))

            # repeat for abstract
            abstract = []
            for word in self.abstract_content[line[0]]:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                abstract.append(self.word_dict[word])
            abstract = abstract[:40]
            
            if self.news_id[line[0]] not in self.news_abstract_dict:
                self.news_abstract_dict[self.news_id[line[0]]] = abstract + [0] * (40 - len(abstract))
            # repeat for entities
            entity = []
            for d in json.loads(line[6]):
                if d['WikidataId'] not in self.entity_dict:
                    self.entity_dict[d['WikidataId']] = len(self.entity_dict)
                entity.append(self.entity_dict[d['WikidataId']])
            for d in json.loads(line[7]):
                if d['WikidataId'] not in self.entity_dict:
                    self.entity_dict[d['WikidataId']] = len(self.entity_dict)
                entity.append(self.entity_dict[d['WikidataId']])
            entity = entity[:5]
            if self.news_id[line[0]] not in self.news_entity_dict:
                self.news_entity_dict[self.news_id[line[0]]] = entity + [0] * (5 - len(entity))
            
    # read the entity embeddings and generate a matrix
    def generate_entity_matrix(self):
        print('generate entity matrix start')
        entity_embed = {}
        # Derive entity embedding paths dynamically from dataset_dir (self.file1 parent)
        dataset_dir = os.path.dirname(os.path.dirname(self.file1))
        train_entity_path = os.path.join(dataset_dir, 'MINDsmall_train', 'entity_embedding.vec')
        dev_entity_path   = os.path.join(dataset_dir, 'MINDsmall_dev',   'entity_embedding.vec')
        for path in (train_entity_path, dev_entity_path):
            with open(path, 'r') as ef:
                for line in ef.readlines():
                    line = line.strip().split('\t')
                    if line[0] not in self.entity_dict:
                        self.entity_dict[line[0]] = len(self.entity_dict)
                    eid = self.entity_dict[line[0]]
                    if eid not in entity_embed:
                        entity_embed[eid] = np.array([float(i) for i in line[1:]])

        self.entity_matrix = [0] * len(self.entity_dict)
        for k, v in self.entity_dict.items():
            if k in entity_embed:
                self.entity_matrix_dict[k] = entity_embed[k]
            else:
                self.entity_matrix_dict[k] = np.zeros(100, dtype='float32')

        self.entity_matrix = torch.FloatTensor(np.array(list(self.entity_matrix_dict.values()), dtype='float32'))
        print('generate entity matrix finished')
        return self.entity_matrix

# acts as a wrapper for the process_news function
    # processes news from 2 files - small_train and small_dev
    # extracts the title and abstract of every article in these files
    def process_train_val_news(self):
        print ('process news start')
        self.process_news(self.file1)
        self.process_news(self.file2)
        self.news_title = np.array(list(self.news_title_dict.values()), dtype = 'int32')
        self.news_abstract = np.array(list(self.news_abstract_dict.values()), dtype = 'int32')
        self.news_entity = np.array(list(self.news_entity_dict.values()), dtype = 'int32')
        print ('news_title.shape: ', self.news_title.shape, 'news_abstract.shape: ', self.news_abstract.shape)
        print ('process news finished')
        #return self.news_title, self.news_abstract, self.news_entity
        #return self.news_title, self.news_entity
        return self.news_title, self.news_abstract

    # to generate user history based on the dev_beh and train_beh
    def generate_user_his(self):
        f3 = open(self.file3)
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue
            
            # extract click history for every user entry
            # padded to max len 50
            click_his_complete = [self.news_id[index] for index in line[3].split()]
            click_his_pad = [self.news_id[index] for index in line[3].split()][:50]
            click_his_pad = click_his_pad + [0] * (50 - len(click_his_pad))

            # if a new user is encountered
            # create an index for the user
            if line[1] not in self.userid_dict:
                self.userid_dict[line[1]] = len(self.userid_dict)

            # add user id
            # complete and padded historys
            if self.userid_dict[line[1]] not in self.user_his_pad:
                self.user_his_pad[self.userid_dict[line[1]]] = click_his_pad
                self.user_his_complete[self.userid_dict[line[1]]] = click_his_complete
        f3.close()

        f4 = open(self.file4)
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            click_his_complete = [self.news_id[index] for index in line[3].split()]
            click_his_pad = [self.news_id[index] for index in line[3].split()][:50]
            click_his_pad = click_his_pad + [0] * (50 - len(click_his_pad))

            if line[1] not in self.userid_dict:
                self.userid_dict[line[1]] = len(self.userid_dict)

            if self.userid_dict[line[1]] not in self.user_his_pad:
                self.user_his_pad[self.userid_dict[line[1]]] = click_his_pad
                self.user_his_complete[self.userid_dict[line[1]]] = click_his_complete
        f4.close()
        return self.user_his_pad

    # --- Popularity Debiasing: compute popularity pools ---
    # Call this BEFORE pre_train_behaviors().
    # Splits all known articles into popular / non-popular sets based on
    # the 80th-percentile click-count threshold per (sub)category.
    def compute_popularity(self):
        print('compute popularity start')
        MIN_TOPIC_SIZE = 5

        # Step 1: Count clicks per article from training behaviors
        click_count = {}
        with open(self.file3) as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                for item in parts[4].split():
                    nid_str, label = item.split('-')
                    if int(label) == 1 and nid_str in self.news_id:
                        nid = self.news_id[nid_str]
                        click_count[nid] = click_count.get(nid, 0) + 1

        # Step 2: Determine popularity threshold (80th percentile)
        all_nids    = list(self.news_id.values())
        all_counts  = [click_count.get(n, 0) for n in all_nids]
        threshold   = float(np.percentile(all_counts, 80)) if all_counts else 0.0
        threshold   = max(threshold, 1.0)  # avoid zero threshold on sparse datasets

        self.popular_set    = {n for n in all_nids if click_count.get(n, 0) >= threshold}
        self.nonpopular_set = {n for n in all_nids if click_count.get(n, 0) <  threshold}

        # Build category_pool for echo-chamber fallback (category -> [nids])
        for nid in all_nids:
            cat = self.news_category.get(nid, 'unknown')
            self.category_pool.setdefault(cat, []).append(nid)

        # Step 3: Build per-topic pools with subcategory-first, category fallback
        subcategory_to_news = {}
        for nid in all_nids:
            sub = self.news_subcategory.get(nid, 'unknown')
            subcategory_to_news.setdefault(sub, []).append(nid)

        def get_topic_key(nid):
            sub = self.news_subcategory.get(nid, 'unknown')
            if len(subcategory_to_news.get(sub, [])) >= MIN_TOPIC_SIZE:
                return sub
            return self.news_category.get(nid, 'unknown')

        topic_to_all = {}
        for nid in all_nids:
            key = get_topic_key(nid)
            topic_to_all.setdefault(key, []).append(nid)

        self._get_topic_key  = get_topic_key
        self._topic_to_all   = topic_to_all
        self._all_topic_keys = list(topic_to_all.keys())

        for key, nids in topic_to_all.items():
            pop_in_topic   = [n for n in nids if n in self.popular_set]
            unpop_in_topic = [n for n in nids if n in self.nonpopular_set]
            if pop_in_topic:
                self.popular_pool[key]    = pop_in_topic
            if unpop_in_topic:
                self.nonpopular_pool[key] = unpop_in_topic

        skippable = sum(1 for k in topic_to_all
                        if k not in self.popular_pool or k not in self.nonpopular_pool)
        print(f'  topics with valid pop+unpop pools: {len(topic_to_all) - skippable}/{len(topic_to_all)}')
        print('compute popularity finished')

    # --- Echo-Chamber Debiasing: pre-compute augmented user histories ---
    # For each user, replaces non-dominant-interest articles with articles from
    # the dominant category (up to echo_threshold fraction).
    def generate_echo_user_his(self, echo_threshold=0.85):
        print('generate echo user his start')
        for user_id, history in self.user_his_pad.items():
            real_ids    = [nid for nid in history if nid != 0]
            padding_len = len(history) - len(real_ids)

            if len(real_ids) == 0:
                self.user_his_echo_pad[user_id] = history[:]
                continue

            category_count = {}
            for nid in real_ids:
                cat = self.newsid_topic.get(nid, 'NULL')
                if cat != 'NULL':
                    category_count[cat] = category_count.get(cat, 0) + 1

            if not category_count:
                self.user_his_echo_pad[user_id] = history[:]
                continue

            dominant_cat          = max(category_count, key=category_count.get)
            real_len              = len(real_ids)
            target_dominant_count = max(int(real_len * echo_threshold), 1)

            dominant_ids    = [nid for nid in real_ids if self.newsid_topic.get(nid, 'NULL') == dominant_cat]
            nondominant_ids = [nid for nid in real_ids if self.newsid_topic.get(nid, 'NULL') != dominant_cat]
            replacements_needed = max(0, target_dominant_count - len(dominant_ids))

            read_set     = set(real_ids)
            pool_unread  = [nid for nid in self.category_pool.get(dominant_cat, [])
                            if nid not in read_set]
            random.shuffle(pool_unread)
            pool_duplicates = dominant_ids[:]

            replacement_articles = []
            pool_idx = dup_idx = 0
            for _ in range(replacements_needed):
                if pool_idx < len(pool_unread):
                    replacement_articles.append(pool_unread[pool_idx]); pool_idx += 1
                elif pool_duplicates:
                    replacement_articles.append(pool_duplicates[dup_idx % len(pool_duplicates)]); dup_idx += 1
                else:
                    break

            random.shuffle(nondominant_ids)
            slots_to_replace = nondominant_ids[:replacements_needed]
            replace_map      = dict(zip(slots_to_replace, replacement_articles))
            augmented_real   = [replace_map.get(nid, nid) for nid in real_ids]
            self.user_his_echo_pad[user_id] = augmented_real + [0] * padding_len

        print('generate echo user his finished')
        return self.user_his_echo_pad

    # 处理训练集数据
    def pre_train_behaviors(self):
        print('reset train variables')
        self.train_candidate = []
        self.train_label     = []
        self.train_user_his  = []
        self.train_user      = []
        self.train_pop       = []
        self.train_unpop     = []
        self.train_diff      = []

        print('process train behaviors start')
        _has_pop_pools = bool(self.popular_pool) and bool(self.nonpopular_pool)
        if not _has_pop_pools:
            print('  [WARNING] compute_popularity() not called. Popularity triplets will all be sentinel (0).')

        f3 = open(self.file3)
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            for doc in p_doc:
                neg_doc = self.newsample(n_doc, self.npratio1)
                neg_doc.append(doc)
                candidate_label = [0] * self.npratio1 + [1]
                candidate_order = list(range(self.npratio1 + 1))
                random.shuffle(candidate_order)
                candidate_shuffle      = []
                candidate_label_shuffle = []
                for i in candidate_order:
                    candidate_shuffle.append(neg_doc[i])
                    candidate_label_shuffle.append(candidate_label[i])
                self.train_candidate.append(candidate_shuffle)
                self.train_label.append(candidate_label_shuffle)
                self.train_user.append(self.userid_dict[line[1]])

                # --- Popularity triplet for this positive sample ---
                pop_id = unpop_id = diff_id = 0  # 0 = sentinel (NULL article)
                if _has_pop_pools and self._get_topic_key is not None:
                    topic_key  = self._get_topic_key(doc)
                    pop_pool   = self.popular_pool.get(topic_key, [])
                    unpop_pool = self.nonpopular_pool.get(topic_key, [])
                    if pop_pool and unpop_pool:
                        pop_id   = random.choice(pop_pool)
                        unpop_id = random.choice(unpop_pool)
                        diff_candidates = [k for k in self._all_topic_keys if k != topic_key]
                        if diff_candidates:
                            diff_topic_key = random.choice(diff_candidates)
                            diff_id = random.choice(self._topic_to_all[diff_topic_key])
                self.train_pop.append(pop_id)
                self.train_unpop.append(unpop_id)
                self.train_diff.append(diff_id)

        self.train_candidate = torch.LongTensor(np.array(self.train_candidate, dtype='int32'))
        self.train_label     = torch.FloatTensor(np.array(self.train_label,     dtype='int32'))
        self.train_user      = torch.LongTensor(np.array(self.train_user,       dtype='int32'))
        self.train_pop       = torch.LongTensor(np.array(self.train_pop,        dtype='int32'))
        self.train_unpop     = torch.LongTensor(np.array(self.train_unpop,      dtype='int32'))
        self.train_diff      = torch.LongTensor(np.array(self.train_diff,       dtype='int32'))

        print('train_candidate.size: ', self.train_candidate.size())
        print('train_label.size:',      self.train_label.size())
        print('train_user.size:',       self.train_user.size())
        print('train_pop.size:',        self.train_pop.size())
        print('process train behaviors finished')
        return [self.train_candidate, self.train_user, self.train_label,
                self.train_pop, self.train_unpop, self.train_diff]


    # 处理验证集数据
    def pre_val_behaviors(self, file):
        print('process val behaviors start')

        f4 = open(file)
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            sess_index = []
            # at every line append the number of validation candidate articles seen so far
            # for any given user's entry in the file, append it's positive and negative articles
            # at every moment, append the length of the number of candidate articles seen so far
            sess_index.append(len(self.val_candidate))
            for i in p_doc:
                # for every postiive article append it to the candidate list - news id
                self.val_candidate.append(i)
                # append 1 for the label
                self.val_label.append(1)
                # append the user id
                # for every news id a user id gets appended
                self.val_user.append(self.userid_dict[line[1]])
            
            # do the same for the negative articles
            for i in n_doc:
                self.val_candidate.append(i)
                self.val_label.append(0)
                self.val_user.append(self.userid_dict[line[1]])

            # number of candidate articles for the given user
            # to know how many more candidate articles were added
            # so val_index keeps the starting and final index to see how many canddiate articles were added for a user
            sess_index.append(len(self.val_candidate))
            self.val_index.append(sess_index)

        self.val_candidate = np.array(self.val_candidate, dtype='int32')
        self.val_label = torch.FloatTensor(np.array(self.val_label, dtype='int32'))
        self.val_user = torch.LongTensor(np.array(self.val_user, dtype = 'int32'))

        print('val_candidate.shape: ', self.val_candidate.shape) # candidate articl ids
        print('val_label.size: ', self.val_label.size()) # labels
        print ('val_user.size:', self.val_user.size()) # user ids #2658091
        print('len(val_index): ', len(self.val_index)) # number of candidate articles for that user #70938

        # removed self.val_index[:900]
        print('process val behaviors finished')
        return [self.val_candidate, self.val_user, self.val_label, self.val_index]

    def pre_test_behaviors(self, file):
        print('process test behaviors start')

        f4 = open(file)
        lines = f4.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            p_doc, n_doc = [], []
            for i in line[4].split():
                p_doc.append(self.news_id[i.split('-')[0]])

            sess_index = []
            sess_index.append(len(self.test_candidate))
            for i in p_doc:
                self.test_candidate.append(i)
                self.test_label.append(1)
                self.test_user.append(self.userid_dict[line[1]])

            sess_index.append(len(self.test_candidate))
            self.test_index.append(sess_index)

        self.test_candidate = np.array(self.test_candidate, dtype='int32')
        self.test_label = torch.FloatTensor(np.array(self.test_label, dtype='int32'))
        self.test_user = torch.LongTensor(np.array(self.test_user, dtype = 'int32'))

        print('test_candidate.shape: ', self.test_candidate.shape)
        print('test_label.size: ', self.test_label.size())
        print ('test_user.size:', self.test_user.size())
        print('len(test_index): ', len(self.test_index))

        print('process test behaviors finished')
        return [self.test_candidate, self.test_user, self.test_label, self.test_index]
            

    # 加载glove预训练模型
    # loads the glove embeddings and creates an embedding matrix for the words in self.word_dict 
    # glove - global vectors for word representation - unsupervised learning algorithm for obtaining vector repr
    # used to capture the semantic relationships between words and based on their co-occurrence in the large corpus of text

    # for each line in the glove file, extracts the word and corresponding vector checking if word exists in word_dict
    # populate the embedding matrix
    # handle missing mebeddings using mean and covariance of vectors
    def load_glove(self):
        print ('load glove start')

        f = open(self.file5, encoding = 'utf-8')
        lines = f.readlines()
        for line in lines:
            if len(line) == 0:
                continue
            line = line.strip().split()
            if len(line) != 301:
                continue
            word = line[0].encode('utf-8').decode()
            if word not in self.word_dict:
                continue
            if len(word) != 0:
                vec = [float(x) for x in line[1:]]
                self.embedding_dict[word] = vec

        self.embedding_matrix = [0] * len(self.word_dict)
        cand = []
        for k, v in self.embedding_dict.items():
            self.embedding_matrix[self.word_dict[k]] = np.array(v, dtype='float32')
            cand.append(self.embedding_matrix[self.word_dict[k]])

        cand = np.array(cand, dtype='float32')
        mu = np.mean(cand, axis=0)
        Sigma = np.cov(cand.T)
        norm = np.random.multivariate_normal(mu, Sigma, 1)
        for i in range(len(self.embedding_matrix)):
            if type(self.embedding_matrix[i]) == int:
                self.embedding_matrix[i] = np.reshape(norm, 300)

        self.embedding_matrix[0] = np.zeros(300, dtype='float32')
        self.embedding_matrix = torch.FloatTensor(np.array(self.embedding_matrix, dtype='float32'))

        print('embedding_matrix.size: ', self.embedding_matrix.size())
        print('load glove process finished')

        return self.embedding_matrix

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np

if __name__ == '__main__':
    
    file1 = 'MINDlarge_train/news.tsv'
    file2 = 'MINDlarge_dev/news.tsv'
    file3 = 'MINDlarge_train/behaviors.tsv'
    file4 = 'MINDlarge_dev/behaviors.tsv'
    file5 = 'glove.840B.300d.txt'

    data_module = DataProcess(file1, file2, file3, file4, file5)
    news_title, news_abstract = data_module.process_train_val_news()
    news_title, news_abstract = torch.LongTensor(news_title), torch.LongTensor(news_abstract)

    [train_candidate, train_label, train_user] = data_module.pre_train_behaviors()
    train_dataset = Data.TensorDataset(train_candidate, train_label, train_user)
    train_loader = Data.DataLoader(dataset = train_dataset, batch_size = 900, shuffle = True, num_workers = 2)

    [val_candidate,val_user, val_label, val_index] = data_module.pre_val_behaviors()
    
    user_his_complete = data_module.user_his_complete
    user_his_pad = torch.LongTensor(np.array(list(data_module.user_his_pad.values()), dtype = 'int32'))
    print (user_his_pad.size(), len(user_his_complete))

    data, user, news = [], [], []
    for u, h in user_his_complete.items():
        for n in h:
            data.append(1)
            user.append(u)
            news.append(n)
    print (len(data), len(data_module.userid_dict), len(data_module.news_id))

    import scipy.sparse as sp
    u_n = sp.csr_matrix((data, (user, news)), shape = (len(data_module.userid_dict), len(data_module.news_id)))
    #print (spm)

    u_u = sp.coo_matrix(u_n.dot(u_n.transpose()))
    #sp.save_npz('./large_user_adj.npz', u_u)
    
    import pickle
    #user_adj = sp.load_npz('./large_user_adj.npz')
    data, row, col = u_u.data, u_u.row, u_u.col
    print (user_adj.shape, type(data), type(row), len(row))
    user_adj_dic = {}
    user_adj = {0: [0] * 3}

    for i in range(len(row)):
        if row[i] not in user_adj_dic:
            user_adj_dic[row[i]] = []
        user_adj_dic[row[i]].append(col[i])
    for k, v in user_adj_dic.items():
        if len(v) > 3:
            user_adj[k] = random.sample(v, 3)
        else:
            user_adj[k] = v + [0] * (3 - len(v))
    user_adj = np.array(list(user_adj.values()), dtype = 'int32')
    print (len(user_adj_dic), user_adj.shape)

    f1 = open('large_user_adj.pkl', 'wb')
    pickle.dump(user_adj, f1)
    f1.close()
    
    f1 = open('large_user_adj.pkl',  'rb')
    user_adj_load = pickle.load(f1)
    user_adj_load = torch.LongTensor(user_adj_load)
    print (user_adj_load.size())
'''