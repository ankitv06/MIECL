from re import S
from torch._C import dtype
from nltk.tokenize import word_tokenize
import numpy as np
import random
import torch
import json

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class DataProcess():
    def __init__(self, file1, file2, file3, file4, file5, file6):
        self.file1 = file1
        self.file2 = file2
        self.file3 = file3
        self.file4 = file4
        self.file5 = file5
        self.file6 = file6

        self.news_id = {'NULL': 0}  # {'NULL': 0, 'N46466': 1}
        self.title_content = {}  # {'N46466': ['the', 'brands', 'queen', 'elizabeth', ',', 'prince', 'charles', ',', 'and', 'prince', 'philip', 'swear', 'by']}
        self.abstract_content = {}
        self.word_dict = {'PADDING': 0}
        self.entity_dict = {'PADDING': 0}
        self.news_title_dict = {'0': [0] * 20}  # {'1': [1, 2, ……, 30]}
        self.news_abstract_dict = {'0': [0] * 40}
        self.news_entity_dict = {'0': [0] * 5}
        self.newsid_topic = {0: 'NULL'}
        self.entity_news = {}
        self.entity_matrix_dict = {0: np.zeros(100, dtype='float32')}
        self.embedding_dict = {}

        self.userid_dict = {'NULL': 0}
        self.npratio1 = 4
        self.npratio2 = 50
        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.user_his_pad = {0: [0] * 50, }
        self.user_his_complete = {0: [], }
        
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

        # --- Popularity Debiased Augmentation ---
        # Maps news int_id -> category string and -> subcategory string
        self.news_category   = {0: 'NULL'}   # {news_int_id: category}
        self.news_subcategory = {0: 'NULL'}  # {news_int_id: subcategory}
        # Maps topic string -> list of news int_ids
        self.subcategory_to_news = {}        # {subcategory: [news_int_ids]}
        self.category_to_news    = {}        # {category:    [news_int_ids]}
        # Click count per article (populated in compute_popularity)
        self.news_click_count = {}           # {news_int_id: click_count}
        # Sets of popular / non-popular article int_ids
        self.popular_set    = set()
        self.nonpopular_set = set()
        # Per-topic pools after popularity classification
        # Each maps topic_key -> list of news_int_ids
        self.popular_pool    = {}            # {topic_key: [popular ids in that topic]}
        self.nonpopular_pool = {}            # {topic_key: [non-popular ids in that topic]}
        # Augmentation triplets (parallel to train_candidate)
        self.train_pop   = []               # popular article id per training sample
        self.train_unpop = []               # non-popular article id per training sample
        self.train_diff  = []               # different-topic article id per training sample
    
    # random sample of a specified size is obtained from the array
    # if npratio is lesser than len(array), samples directly from array
    # else creates a repeated version of the array for sampling
    def newsample(self, array, npratio):
        if npratio > len(array):
            return random.sample(array*(npratio // len(array) + 1), npratio)
        else:
            return random.sample(array, npratio)


    # 处理新闻数据

    # read and process news from a file containing news data: generates embeddings for title, abstract and entity
    # Also parses category (line[1]) and subcategory (line[2]) for popularity debiasing
    def process_news(self, file):
        f = open(file, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            self.title_content[line[0]] = word_tokenize((line[3]).lower())
            self.abstract_content[line[0]] = word_tokenize((line[4]).lower())
            if line[0] not in self.news_id:
                # assigns a new id for the news article incase it does not exist already
                self.news_id[line[0]] = len(self.news_id)

            nid = self.news_id[line[0]]

            # --- Parse category and subcategory (popularity debiasing) ---
            category    = line[1] if len(line) > 1 else 'unknown'
            subcategory = line[2] if len(line) > 2 else 'unknown'
            self.news_category[nid]    = category
            self.news_subcategory[nid] = subcategory
            # Build reverse maps: topic -> list of article ids
            if subcategory not in self.subcategory_to_news:
                self.subcategory_to_news[subcategory] = []
            if nid not in self.subcategory_to_news[subcategory]:
                self.subcategory_to_news[subcategory].append(nid)
            if category not in self.category_to_news:
                self.category_to_news[category] = []
            if nid not in self.category_to_news[category]:
                self.category_to_news[category].append(nid)
            
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
            
    # --- Popularity Debiased Augmentation: compute article popularity from training behaviors ---
    # Reads training behaviors.tsv, counts how many times each article was clicked,
    # then classifies articles into popular (top 15%) and non-popular (bottom 75%) pools.
    # Builds per-topic (subcategory-first, category fallback) pools for pair sampling.
    # MIN_TOPIC_SIZE controls the fallback threshold: subcategories with fewer articles
    # than this value use their parent category instead.
    def compute_popularity(self):
        print('compute popularity start')
        MIN_TOPIC_SIZE = 20    # subcategory fallback threshold
        POPULAR_PCTILE  = 85   # top 15%  -> popular   (100 - 15 = 85th percentile cutoff)
        UNPOPULAR_PCTILE = 75  # bottom 75% -> non-popular

        # --- Step 1: Count clicks per article across all training impressions ---
        f = open(self.file3, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if len(line) < 5 or line[4] == '':
                continue
            for item in line[4].split():
                parts = item.split('-')
                if len(parts) != 2:
                    continue
                news_str, label = parts[0], parts[1]
                if label == '1' and news_str in self.news_id:
                    nid = self.news_id[news_str]
                    self.news_click_count[nid] = self.news_click_count.get(nid, 0) + 1
        f.close()

        # Articles that were never clicked get a count of 0
        all_nids = list(self.news_id.values())
        for nid in all_nids:
            if nid not in self.news_click_count:
                self.news_click_count[nid] = 0

        # --- Step 2: Determine popularity thresholds ---
        counts = list(self.news_click_count.values())
        nonzero_counts = [cnt for cnt in counts if cnt > 0]
        
        if nonzero_counts:
            pop_threshold   = float(np.percentile(nonzero_counts, POPULAR_PCTILE))   # >= this -> popular
            unpop_threshold = float(np.percentile(nonzero_counts, UNPOPULAR_PCTILE)) # <= this -> non-popular
        else:
            pop_threshold, unpop_threshold = 1.0, 0.0
            
        print(f'  popular threshold (>= {POPULAR_PCTILE}th pctile of non-zero): {pop_threshold} clicks')
        print(f'  non-popular threshold (<= {UNPOPULAR_PCTILE}th pctile of non-zero): {unpop_threshold} clicks')

        for nid, cnt in self.news_click_count.items():
            if cnt >= pop_threshold:
                self.popular_set.add(nid)
            if cnt <= unpop_threshold:
                self.nonpopular_set.add(nid)
        print(f'  popular articles: {len(self.popular_set)}, non-popular articles: {len(self.nonpopular_set)}')

        # --- Step 3: Build per-topic pools with subcategory-first, category fallback ---
        # Determine which topic key to use for each article
        def get_topic_key(nid):
            sub = self.news_subcategory.get(nid, 'unknown')
            if len(self.subcategory_to_news.get(sub, [])) >= MIN_TOPIC_SIZE:
                return sub
            return self.news_category.get(nid, 'unknown')

        # Group all articles by their resolved topic key
        topic_to_all = {}   # topic_key -> [all nids in that topic]
        for nid in all_nids:
            key = get_topic_key(nid)
            topic_to_all.setdefault(key, []).append(nid)

        # Store the topic key resolver for use during training sample generation
        self._get_topic_key = get_topic_key
        self._topic_to_all  = topic_to_all
        self._all_topic_keys = list(topic_to_all.keys())

        # Split each topic into popular / non-popular sub-pools
        for key, nids in topic_to_all.items():
            pop_in_topic    = [n for n in nids if n in self.popular_set]
            unpop_in_topic  = [n for n in nids if n in self.nonpopular_set]
            if pop_in_topic:
                self.popular_pool[key]    = pop_in_topic
            if unpop_in_topic:
                self.nonpopular_pool[key] = unpop_in_topic

        skippable = sum(1 for k in topic_to_all if k not in self.popular_pool or k not in self.nonpopular_pool)
        print(f'  topics with valid pop+unpop pools: {len(topic_to_all) - skippable}/{len(topic_to_all)}')
        print('compute popularity finished')

    # read the entity embeddings and generate a matrix
    def generate_entity_matrix(self):
        print ('generate entity matrix start')
        entity_embed = {}
        f1 = open('/content/drive/MyDrive/News Recc Code/dataset/MINDsmall_train/entity_embedding.vec', 'r')
        lines1 = f1.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if line[0] not in self.entity_dict:
                self.entity_dict[line[0]] = len(self.entity_dict)
            if self.entity_dict[line[0]] not in entity_embed:
                entity_embed[self.entity_dict[line[0]]] = np.array([float(i) for i in line[1:]])
        f2 = open('/content/drive/MyDrive/News Recc Code/dataset/MINDsmall_dev/entity_embedding.vec', 'r')
        lines2 = f2.readlines()
        for line in lines2:
            line = line.strip().split('\t')
            if line[0] not in self.entity_dict:
                self.entity_dict[line[0]] = len(self.entity_dict)
            if self.entity_dict[line[0]] not in entity_embed:
                entity_embed[self.entity_dict[line[0]]] = np.array([float(i) for i in line[1:]])
        
        # dictionary
        # keys are the embedding ids
        # values - 100-dim embedding from the MIND dataset file
        self.entity_matrix = [0] * len(self.entity_dict)
        for k,v in self.entity_dict.items():
            if k in entity_embed:
                self.entity_matrix_dict[k] = entity_embed[k]
            else:
                self.entity_matrix_dict[k] = np.zeros(100, dtype='float32')
        
        self.entity_matrix = torch.FloatTensor(np.array(list(self.entity_matrix_dict.values()), dtype = 'float32'))
        print ('generate entity matrix finished')
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

            # always overwrite with the latest session so we capture the most complete history
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

            # always overwrite with the latest session so we capture the most complete history
            self.user_his_pad[self.userid_dict[line[1]]] = click_his_pad
            self.user_his_complete[self.userid_dict[line[1]]] = click_his_complete
        f4.close()
        return self.user_his_pad

    # 处理训练集数据
    # aim is to process the training behaviors, generates + and - samples, shuffles and organizes them
    # stores the processed data in tensors
    # prints the size of the tensors and returns them as a list
    def pre_train_behaviors(self):
        print ('reset train variables')
        self.train_candidate = []
        self.train_label = []
        self.train_user_his = []
        self.train_user = []
        self.train_pop   = []
        self.train_unpop = []
        self.train_diff  = []
        
        print ('process train behaviors start')

        # Check whether compute_popularity() has been called; warn if not.
        _has_pop_pools = bool(self.popular_pool) and bool(self.nonpopular_pool)
        if not _has_pop_pools:
            print('  [WARNING] compute_popularity() not called. Augmentation triplets will all be sentinel (0).')

        # splits and processes every line
        f3 = open(self.file3)
        lines = f3.readlines()
        for line in lines:
            line = line.strip().split('\t')
            if line[3] == '':
                continue

            # positive and negative samples
            # 1 - user clicked on the news article (positive sample)
            # 0 - user did not click on the shown article - negative sample
            # add the entries to the respective lists
            p_doc, n_doc = [], []
            for i in line[4].split():
                if int(i.split('-')[1]) == 1:
                    p_doc.append(self.news_id[i.split('-')[0]])
                elif int(i.split('-')[1]) == 0:
                    n_doc.append(self.news_id[i.split('-')[0]])

            # for every positive sample
            # a list of negative samples is created using the newsample method
            # pos_doc is appended to the lsit of neg_doc - combo of + and - samples
            # a candidate_label is created with 'self.npratio1' zeros followed by 1 - represents the labels
                # candidate sample labelling, 1 - +, 0 - -
            # candidate order - list of integers from 0 to self.npratio + 1 - order will be used to shuffle the pos and neg samples
            for doc in p_doc:
                neg_doc = self.newsample(n_doc, self.npratio1)
                neg_doc.append(doc)
                candidate_label = [0] * self.npratio1 + [1]
                candidate_order = list(range(self.npratio1 + 1))
                random.shuffle(candidate_order)
                candidate_shuffle = []
                candidate_label_shuffle = []
                for i in candidate_order:
                    # for every index i in the shuffled order
                # ith lable neg doc is appended to can_shuffle
                # ith label appended to can_label_shuffle
                    candidate_shuffle.append(neg_doc[i])
                    candidate_label_shuffle.append(candidate_label[i])
                self.train_candidate.append(candidate_shuffle)
                self.train_label.append(candidate_label_shuffle)
                self.train_user.append(self.userid_dict[line[1]])

                # --- Popularity Debiased Augmentation: generate triplet for this sample ---
                # doc is the positive (clicked) article for this training sample.
                # We find its topic, then sample one popular and one non-popular article
                # from the same topic, plus one article from a randomly chosen different topic.
                pop_id, unpop_id, diff_id = 0, 0, 0  # sentinel = 0 (NULL article)
                if _has_pop_pools:
                    topic_key = self._get_topic_key(doc)
                    pop_pool   = self.popular_pool.get(topic_key, [])
                    unpop_pool = self.nonpopular_pool.get(topic_key, [])
                    if pop_pool and unpop_pool:
                        pop_id   = random.choice(pop_pool)
                        unpop_id = random.choice(unpop_pool)
                        # Sample a different topic randomly (ensure it differs from topic_key)
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

        print ('train_candidate.size: ', self.train_candidate.size())
        print ('train_label.size:',      self.train_label.size())
        print ('train_user.size:',       self.train_user.size())
        print ('train_pop.size:',        self.train_pop.size())
        print ('train_unpop.size:',      self.train_unpop.size())
        print ('train_diff.size:',       self.train_diff.size())
        print ('process train behaviors finished')
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

        print('val_candidate.shape: ', self.val_candidate.shape) # candidate article ids
        print('val_label.size: ', self.val_label.size()) # labels
        print ('val_user.size:', self.val_user.size()) # user ids
        print('len(val_index): ', len(self.val_index)) # total impressions (e.g. 70938 for MINDsmall)
        
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
                # An empty line is a file artifact; skip it and keep loading.
                # Using `break` here would silently stop loading all remaining
                # GloVe vectors — changed to `continue`.
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