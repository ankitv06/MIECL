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
    parser.add_argument('--dataset_dir', type=str, default='dataset',
                        help='Root folder containing MINDsmall_train/, MINDsmall_dev/, glove/')
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and go straight to validation')
    parser.add_argument('--eval_epochs', type=int, nargs='+', default=[],
                        help='Specific epochs to evaluate (e.g., --eval_epochs 1 20 200). If empty, evaluates all.')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Batch size to use during evaluation to prevent OOM (default 32).')
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

    if not os.path.exists(preserve_dir):
        os.makedirs(preserve_dir)

    file1 = os.path.join(dataset_dir, 'MINDsmall_train/news.tsv')
    file2 = os.path.join(dataset_dir, 'MINDsmall_dev/news.tsv')
    file3 = os.path.join(dataset_dir, 'MINDsmall_train/behaviors.tsv')
    file4 = os.path.join(dataset_dir, 'MINDsmall_dev/behaviors.tsv')
    file5 = os.path.join(dataset_dir, 'glove.840B.300d.txt')
    file6 = 'dummy.txt'

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
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode)
    model = model.to(device)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
 
    #user_adj = []
    #f = open('small_user_nei_sort.txt', 'r', encoding='utf-8')
    #lines = f.readlines()
    #for line in lines:
    #    line = line.strip().split('\t')
    #    user_adj.append([int(i) for i in line])
    #user_adj = torch.LongTensor(np.array(user_adj, dtype = 'int32'))
    #print ('user_adj.size: ', user_adj.size())
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    if not args.eval_only:
        for n_d in range(num_dataset):
            # training loop
            [train_candidate, train_user, train_label] = data_module.pre_train_behaviors()
            train_dataset = Data.TensorDataset(train_candidate, train_user, train_label)
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            
            for n_ep in range(num_epoch):
                acc, all = 0, 0
                t0 = time.time()
                loss_per_epoch = []

                # batches from the training loader
                for step, (train_candidate, train_user, train_label) in enumerate(train_loader):
                    t1 = time.time()
                    candidate_title, his_title, train_label = news_title[train_candidate].to(device), news_title[user_his[train_user]].to(device), train_label.to(device)
                    candidate_title, his_title, train_label = Variable(candidate_title),Variable(his_title), Variable(train_label)
                    candidate_abstract, his_abstract  = news_abstract[train_candidate].to(device), news_abstract[user_his[train_user]].to(device)
                    candidate_abstract, his_abstract  = Variable(candidate_abstract),Variable(his_abstract)
                    print (candidate_title.size(), candidate_abstract.size())
    
                    #neighbor_user = user_adj[train_user]
                    #(neighbor_1, neighbor_2) = torch.split(neighbor_user, 1, dim = 1)
                    #neighbor_1, neighbor_2 = neighbor_1.squeeze(dim = 1), neighbor_2.squeeze(dim = 1)
                    
                    #nei1_title, nei1_abstract  = news_title[user_his[neighbor_1]].cuda(), news_abstract[user_his[neighbor_1]].cuda()
                    #nei1_title, nei1_abstract = Variable(nei1_title), Variable(nei1_abstract)
                    #nei2_title, nei2_abstract  = news_title[user_his[neighbor_2]].cuda(), news_abstract[user_his[neighbor_2]].cuda()
                    #nei2_title, nei2_abstract = Variable(nei2_title), Variable(nei2_abstract)
    
                    model.train()
                    optimizer.zero_grad()
    
                    #predictor_logits, user_infoNCE_logits = model(candidate_title, candidate_abstract, his_title, his_abstract, neighbor_title, neighbor_abstract)
                    predictor_logits, user_infoNCE_logits = model(candidate_title, candidate_abstract, his_title, his_abstract)
                    predictor_loss = criterion(predictor_logits, train_label)
                    
                    if contrastive_mode == 'USER':
                        user_infoNCE_labels = torch.zeros(len(user_infoNCE_logits), dtype=torch.long, device=device)
                        user_infoNCE_loss = F.cross_entropy(user_infoNCE_logits, user_infoNCE_labels)
                        print ('predictor_loss: ', predictor_loss.data.item(), 'user_infoNCE_loss: ', user_infoNCE_loss.data.item())
                        loss = predictor_loss + alpha * user_infoNCE_loss
                    else:
                        print ('predictor_loss: ', predictor_loss.data.item())
                        loss = predictor_loss
                        
                    loss.backward()
                    optimizer.step()
    
                    loss_per_epoch.append(loss.data.item())
                    print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'step: {:04d}'.format(step + 1), 'loss: {:.4f}'.format(np.mean(loss_per_epoch)), 'time: {:.4f}'.format(time.time() - t1))

            torch.save(model.state_dict(), preserve_dir + '/model_{}.pkl'.format(n_d * num_epoch + n_ep + 1))
            print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'time: {:.4f}'.format(time.time() - t0))
        del train_candidate, train_user, train_label
    else:
        print("Skipping training, starting validation from saved checkpoint...")

    print("TRAINING DONE-------------------------------------------------------------------------------------------------------------------------------------------------")
    # validation and evaluation
    torch.cuda.empty_cache()
    
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
    for idx, i in enumerate(val_index):
        i_label = val_label[i[0]: i[1]].data.numpy().tolist()
        truth_file.write(str(idx) + ' ' + '[')
        for item in i_label[:-1]:
            truth_file.write(str(item) + ',')
        truth_file.write(str(i_label[-1]) + ']' + '\n')
    truth_file.flush()
    truth_file.close()

    
    val_dataset = Data.TensorDataset(val_candidate, val_user, val_label)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=args.eval_batch_size, shuffle=False, num_workers=2)

    epochs_to_eval = range(1, num_dataset * num_epoch + 1)
    if args.eval_epochs:
        epochs_to_eval = args.eval_epochs

    for epoch_idx in epochs_to_eval:
        #model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode)
        #loaded_dict = torch.load(preserve_dir + '/model_{}.pkl'.format(n_d + 1))
        #model = nn.DataParallel(model, device_ids = [0])
        #model.state_dict = loaded_dict
        #print (next(model.parameters()).device)

        checkpoint_path = preserve_dir + '/model_{}.pkl'.format(epoch_idx)
        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found:", checkpoint_path)
            continue
            
        model.load_state_dict(torch.load(checkpoint_path))
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

        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(epoch_idx), 'w')
        print ('process predict_file_{} start'.format(epoch_idx))

        cnt = 0
        for idx, i in enumerate(val_index):
            i_score = [item for item in val_score[i[0]: i[1]]]
            i_score_sort = sorted(i_score, reverse=True)
            
            rank = []
            for item in i_score:
                rank.append(i_score_sort.index(item) + 1)
            predict_file.write(str(idx) + ' ' + '[')
            for item in rank[:-1]:
                predict_file.write(str(item) + ',')
            predict_file.write(str(rank[-1]) + ']' + '\n')

        predict_file.flush()
        predict_file.close()
        print ('process predict_file_{} finished'.format(epoch_idx))
        
        print ('calculate {}_th auc/mrr/ndcg start'.format(epoch_idx))
        output_filename = preserve_dir + '/scores_{}.txt'.format(epoch_idx)
        output_file = open(output_filename, 'w')

        truth_file = open(preserve_dir + '/truth.txt', 'r')
        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(epoch_idx), 'r')

        auc, mrr, ndcg, ndcg10 = scoring(truth_file, predict_file)

        output_file.write("AUC:{:.4f}\nMRR:{:.4f}\nnDCG@5:{:.4f}\nnDCG@10:{:.4f}".format(auc, mrr, ndcg, ndcg10))
        output_file.close()
        print ('calculate {}_th auc/mrr/ndcg finished'.format(epoch_idx))
