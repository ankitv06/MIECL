from data_process import DataProcess
from model import Multi_Rep_Predictor
from evaluate import scoring
from torch.utils.data import Subset


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

import time
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
import argparse
import os
import glob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=1)
    parser.add_argument('--num_dataset', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--hid_dim', type=int, default=400)
    parser.add_argument('--num_head', type=int, default=20)
    parser.add_argument('--num_prototype', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--num_negative_sample', type=int, default=3)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--preserve_dir', type=str, default='C:/Users/anany/Desktop/Ananya/2023/Estonia Projects/News Recc/MIECL-master')
    parser.add_argument('--pretrain_method', type=str, default='glove')
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--multi_rep_mode', type=str, default='concat')
    parser.add_argument('--infonce_mode', type=str, default='prototype_self')
    parser.add_argument('--contrastive_mode', type=str, default='USER')
    parser.add_argument('--gnn_mode', type=str, default='nogat')
    parser.add_argument('--agg_mode', type=str, default='soft')
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--glove_path', type=str, default='dataset/glove.840B.300d.txt')
    parser.add_argument('--val_only', action='store_true', help='Skip training and run only validation')
    args = parser.parse_args()

    num_epoch = args.num_epoch
    num_dataset = args.num_dataset
    lr = args.lr
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    hid_dim = args.hid_dim
    num_head = args.num_head
    word_dim = args.word_dim
    num_negative_sample = args.num_negative_sample
    preserve_dir = args.preserve_dir
    pretrain_method = args.pretrain_method
    num_prototype = args.num_prototype
    alpha = args.alpha
    dropout_rate = args.dropout_rate
    multi_rep_mode = args.multi_rep_mode
    infonce_mode = args.infonce_mode
    contrastive_mode = args.contrastive_mode
    gnn_mode = args.gnn_mode
    agg_mode = args.agg_mode
    dataset_dir = args.dataset_dir
    glove_path = args.glove_path
    val_only = args.val_only

    if not os.path.exists(preserve_dir):
        os.makedirs(preserve_dir)

    file1 = os.path.join(dataset_dir, 'MINDsmall_train/news.tsv')
    file2 = os.path.join(dataset_dir, 'MINDsmall_dev/news.tsv')
    file3 = os.path.join(dataset_dir, 'MINDsmall_train/behaviors.tsv')
    file4 = os.path.join(dataset_dir, 'MINDsmall_dev/behaviors.tsv')
    file5 = glove_path
    file6 = 'dummy.txt'

    # Verify files exist
    for f_path in [file1, file2, file3, file4, file5]:
        if not os.path.exists(f_path):
            raise FileNotFoundError(f"Required dataset file not found: {f_path}. Please check your --dataset_dir or --glove_path arguments.")

    data_module = DataProcess(file1, file2, file3, file4, file5, file6)
    news_title, news_abstract = data_module.process_train_val_news()
    news_title, news_abstract = torch.LongTensor(news_title), torch.LongTensor(news_abstract)
    
    entity_matrix = data_module.generate_entity_matrix()
    entity_dim = entity_matrix.size(1)

    word_matrix = None
    if pretrain_method == 'glove':
        word_matrix = data_module.load_glove()

    user_his = data_module.generate_user_his()
    user_his = torch.LongTensor(np.array(list(user_his.values()), dtype = 'int32'))
    print ('num_user: ', len(user_his))

    # --- Echo-Chamber Debiasing: pre-compute augmented user histories ---
    # Only generated when infonce_mode is 'echo_chamber_debiased';
    # for all other modes user_his_echo remains None and is never used.
    user_his_echo = None
    if infonce_mode == 'echo_chamber_debiased':
        print('infonce_mode=echo_chamber_debiased: generating echo-chamber user histories...')
        user_his_echo_raw = data_module.generate_echo_user_his(echo_threshold=0.85)
        user_his_echo = torch.LongTensor(np.array(list(user_his_echo_raw.values()), dtype='int32'))
        print('user_his_echo.size: ', user_his_echo.size())
    # --- End Echo-Chamber Debiasing ---
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode)
    model = model.to(device)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    
    #model.load_state_dict(torch.load('/home/wangshicheng/news_recommendation/Final_edtion/title_abstract_edition/concat_dr0.0_prototype_other_user_nogat_soft_6_3_5_s/model_{}.pkl'.format(i + 1)))
    #model.load_state_dict(torch.load('/home/wangshicheng/news_recommendation/Final_edtion/title_abstract_edition/sgd_other2_10_1_5_l_adam_val_2/model_6.pkl'))
    model = model
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    #optimizer = optim.Adamax(model.parameters(), lr=0.002)
    
    best_epoch = 0
    min_loss = float('inf')

    if not val_only:
        for n_d in range(num_dataset):
            # training loop
            [train_candidate, train_user, train_label] = data_module.pre_train_behaviors()
            train_dataset = Data.TensorDataset(train_candidate, train_user, train_label)
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            # --- Training loop (echo_chamber_debiased mode adds echo history per batch) ---
            for n_ep in range(num_epoch):
                acc, all = 0, 0
                t0 = time.time()
                loss_per_epoch = []

                # batches from the training loader
                # news titles and abstracts are obtained based on user behavior
                # model set to training mode + gradients set to 0
                # model saved after every epoch
                for step, (train_candidate, train_user, train_label) in enumerate(train_loader):
                    t1 = time.time()
                    candidate_title  = news_title[train_candidate].to(device)
                    his_title        = news_title[user_his[train_user]].to(device)
                    candidate_title, his_title, train_label = Variable(candidate_title), Variable(his_title), Variable(train_label).to(device)
                    candidate_abstract = news_abstract[train_candidate].to(device)
                    his_abstract       = news_abstract[user_his[train_user]].to(device)
                    candidate_abstract, his_abstract = Variable(candidate_abstract), Variable(his_abstract)

                    # --- Echo-Chamber Debiasing: fetch pre-computed echo histories for this batch ---
                    # echo_his_title/abstract have the same shape as his_title/abstract: [batch, 50, word_len]
                    # They are only passed to the model when infonce_mode == 'echo_chamber_debiased';
                    # for all other modes they remain None and the model ignores them.
                    echo_his_title    = None
                    echo_his_abstract = None
                    if infonce_mode == 'echo_chamber_debiased':
                        echo_his_title    = Variable(news_title[user_his_echo[train_user]]).to(device)
                        echo_his_abstract = Variable(news_abstract[user_his_echo[train_user]]).to(device)
                    # --- End Echo-Chamber Debiasing ---

                    model.train()
                    optimizer.zero_grad()

                    predictor_logits, user_infoNCE_logits = model(
                        candidate_title, candidate_abstract,
                        his_title, his_abstract,
                        echo_his_title, echo_his_abstract   # None for non-echo modes
                    )
                    predictor_loss = criterion(predictor_logits, train_label)

                    if contrastive_mode == 'USER':
                        # labels are zeros: index 0 in logits is always the positive (I_k)
                        # for echo_chamber_debiased, logits shape is [B*(K-1), 2];
                        # for prototype_self, logits shape is [B, 2].
                        # F.cross_entropy handles both via its built-in mean reduction.
                        user_infoNCE_labels = torch.zeros(len(user_infoNCE_logits), dtype=torch.long, device=device)
                        user_infoNCE_loss   = F.cross_entropy(user_infoNCE_logits, user_infoNCE_labels)
                        print('predictor_loss: ', predictor_loss.data.item(), 'user_infoNCE_loss: ', user_infoNCE_loss.data.item())
                        loss = predictor_loss + alpha * user_infoNCE_loss
                    else:
                        print('predictor_loss: ', predictor_loss.data.item())
                        loss = predictor_loss

                    loss.backward()
                    optimizer.step()

                    loss_per_epoch.append(loss.data.item())
                    print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1),
                          'step: {:04d}'.format(step + 1),
                          'loss: {:.4f}'.format(np.mean(loss_per_epoch)),
                          'time: {:.4f}'.format(time.time() - t1))

                torch.save(model.state_dict(), preserve_dir + '/model_{}.pkl'.format(n_d * num_epoch + n_ep + 1))
                print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'time: {:.4f}'.format(time.time() - t0))
            del train_candidate, train_user, train_label
    else:
        print("Skipping training, starting validation from saved checkpoint...")
    print("TRAINING DONE-------------------------------------------------------------------------------------------------------------------------------------------------")
    # validation and evaluation
    
    # cand articles, user ids, labels (0/1), number of candidate articles for that user
    [val_candidate, val_user, val_label, val_index] = data_module.pre_val_behaviors(file4)
    val_candidate = torch.LongTensor(val_candidate)


    f = open(preserve_dir + '/val_label.pkl', 'wb')
    # dump all the labels in the file
    pickle.dump(val_label, f)
    f.close()
    f = open(preserve_dir + '/val_index.pkl', 'wb')
    pickle.dump(val_index, f)
    # dump number of candidate articles for every user
    f.close()    

    truth_file = open(preserve_dir + '/truth.txt', 'w')
    # number of val users
    print(len(val_index))
    for i in val_index:
        # val index contains number of candidate articles
        # val labels contains the labels adn convert to list
        # if index =3, select the first 3 labels
        i_label = val_label[i[0]: i[1]].data.numpy().tolist()
        # this is just indexing - 0,1,2..
        truth_file.write(str(val_index.index(i)) + ' ' + '[')
        for item in i_label[:-1]:
            # write down the labels
            truth_file.write(str(item) + ',')
        # close the bracket
        truth_file.write(str(i_label[-1]) + ']' + '\n')
    truth_file.flush()
    truth_file.close()

    
    val_dataset = Data.TensorDataset(val_candidate, val_user, val_label)
    #val_loader = Data.DataLoader(dataset=val_dataset, batch_size=batch_size * 3, shuffle=False, num_workers=2)

    subset_indices = range(32760)  # Choose the indices of the entries you want to include
    subset_dataset = Subset(val_dataset, subset_indices)

    # Create a new DataLoader with the subset dataset
    subset_loader = Data.DataLoader(dataset=subset_dataset, batch_size=batch_size * 3, shuffle=False, num_workers=2)
    val_loader = subset_loader

    #val_candidate = np.array_split(val_candidate, 8000)     # [7600, 1800] , [11400, 1200], [22800, 600], [15200, 900]
    #val_user = np.array_split(val_user, 8000)       # [9120, 1500] , [34200, 400], [30400, 450]
    # [90, 26600] [400, 6600]

    for n_d in range(num_dataset * num_epoch):
        #model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode)
        #loaded_dict = torch.load(preserve_dir + '/model_{}.pkl'.format(n_d + 1))
        #model = nn.DataParallel(model, device_ids = [0])
        #model.state_dict = loaded_dict
        #print (next(model.parameters()).device)

        model.load_state_dict(torch.load(preserve_dir + '/model_{}.pkl'.format(n_d + 1)))
        model = model
        model.eval()
        val_score = []
        t = time.time()        

        with torch.no_grad():
            # score evaluation done batchwise
            # then val index is used to extract the scores relevant to the user
            for step, (val_candidate, val_user, val_label) in enumerate(val_loader):
            #for i in range(len(val_candidate)):
                t1 = time.time()
                print ('index_of_batch_valdataset: ', step)
                #print ('index_of_batch_valdataset: ', i)

                #temp_candidate_title, temp_his_title = news_title[torch.LongTensor(val_candidate[i])].unsqueeze(dim = 1).cuda(), news_title[user_his[torch.LongTensor(val_user[i])]].cuda()
                candidate_title, his_title = news_title[val_candidate].unsqueeze(dim = 1).to(device), news_title[user_his[val_user]].to(device)
                candidate_title, his_title = Variable(candidate_title), Variable(his_title)
                candidate_abstract, his_abstract = news_abstract[val_candidate].unsqueeze(dim = 1).to(device), news_abstract[user_his[val_user]].to(device)
                candidate_abstract, his_abstract = Variable(candidate_abstract), Variable(his_abstract)
                print (candidate_title.size(), his_title.size(), candidate_abstract.size(), his_abstract.size())

                #neighbor_user = user_adj[val_user]
                #neighbor_1, neighbor_2 = torch.split(neighbor_user, 1, dim = 1)
                #neighbor_1, neighbor_2 = neighbor_1.squeeze(dim = 1), neighbor_2.squeeze(dim = 1)
                
                #nei1_title, nei1_abstract  = news_title[user_his[neighbor_1]].cuda(), news_abstract[user_his[neighbor_1]].cuda()
                #nei1_title, nei1_abstract = Variable(nei1_title), Variable(nei1_abstract)
                #nei2_title, nei2_abstract  = news_title[user_his[neighbor_2]].cuda(), news_abstract[user_his[neighbor_2]].cuda()
                #nei2_title, nei2_abstract = Variable(nei2_title), Variable(nei2_abstract)
                #print (nei1_title.size(), nei1_abstract.size(), nei2_title.size(), nei2_abstract.size())

                #neighbor_user = user_adj[torch.LongTensor(val_user[i])].reshape(-1, 1)
                #neighbor_user = user_adj[val_user].reshape(-1, 1)
                #neighbor_title, neighbor_abstract  = news_title[user_his[neighbor_user]].squeeze(dim = 1).cuda(), news_abstract[user_his[neighbor_user]].squeeze(dim = 1).cuda()
                #neighbor_title, neighbor_abstract = Variable(neighbor_title), Variable(neighbor_abstract)
                #print (neighbor_title.size(), neighbor_abstract.size())

                predictor_logits, _ = model(candidate_title, candidate_abstract, his_title, his_abstract)
                # prob of clicking on that article
                score = torch.sigmoid(predictor_logits).cpu().data.numpy()
                val_score = val_score + score.tolist()
            print('val_time: {:.4f}'.format(time.time() - t), 'val_score.length: ', len(val_score))
        f = open(preserve_dir + '/val_score_{}.pkl'.format(n_d + 1), 'wb')
        pickle.dump(val_score, f)
        f.close()

        #f1 = open(preserve_dir + '/val_index.pkl', 'rb')
        #f2 = open(preserve_dir + '/val_score_{}.pkl'.format(n_d + 1), 'rb')
        #f3 = open(preserve_dir + '/val_label.pkl', 'rb')

        #val_index = pickle.load(f1)
        #val_score = pickle.load(f2)
        #val_label = pickle.load(f3)

        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(n_d + 1), 'w')
        print ('process predict_file_{} start'.format(n_d + 1))

        # every term in val index represents the number of candidate articles associated with every user
        #print('val score: ', val_score)
    # )e one pair of indices - a list
        cnt = 0
        for i in val_index:
            # extract the list of scores for all the correponding news artciles using the obtained indices
            i_score = [item for item in val_score[i[0]: i[1]]]
            # sort the scores
            i_score_sort = sorted(i_score, reverse=True)
            
            rank = []
            for item in i_score:
                # obtain the rank for the articles based on their position in the sorted score list
                rank.append(i_score_sort.index(item) + 1)
            predict_file.write(str(val_index.index(i)) + ' ' + '[')
            print(rank)
            for item in rank[:-1]:
                predict_file.write(str(item) + ',')
            predict_file.write(str(rank[-1]) + ']' + '\n')

        predict_file.flush()
        predict_file.close()
        print ('process predict_file_{} finished'.format(n_d + 1))
        
        print ('calculate {}_th auc/mrr/ndcg start'.format(n_d + 1))
        output_filename = preserve_dir + '/scores_{}.txt'.format(n_d + 1)
        output_file = open(output_filename, 'w')

        truth_file = open(preserve_dir + '/truth.txt', 'r')
        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(n_d + 1), 'r')

        auc, mrr, ndcg, ndcg10 = scoring(truth_file, predict_file)

        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))
        output_file.close()
        print ('calculate {}_th auc/mrr/ndcg finished'.format(n_d + 1))
        
