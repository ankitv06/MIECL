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
import csv


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
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.07)
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
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training and only run evaluation on saved checkpoints.')
    parser.add_argument('--eval_epoch', type=int, default=-1,
                        help='Specific epoch to evaluate (e.g., 200). If -1, evaluates all.')
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
    beta  = args.beta
    temperature = args.temperature
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
    # Always generated when contrastive_mode is 'USER' — both proto and echo losses are
    # computed during every training step regardless of infonce_mode (which is kept as a label).
    user_his_echo = None
    if contrastive_mode == 'USER':
        print('Generating echo-chamber user histories for contrastive training...')
        user_his_echo_raw = data_module.generate_echo_user_his(echo_threshold=0.85)
        user_his_echo = torch.LongTensor(np.array(list(user_his_echo_raw.values()), dtype='int32'))
        print('user_his_echo.size: ', user_his_echo.size())
    # --- End Echo-Chamber Debiasing ---
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode, temperature=temperature)
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

    if not args.eval_only and not val_only:
        for n_d in range(num_dataset):
            # training loop
            [train_candidate, train_user, train_label] = data_module.pre_train_behaviors()
            train_dataset = Data.TensorDataset(train_candidate, train_user, train_label)
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            # --- Training loop (echo_chamber_debiased mode adds echo history per batch) ---
            for n_ep in range(num_epoch):
                curr_epoch = n_d * num_epoch + n_ep + 1
                log_csv_path = os.path.join(preserve_dir, f'epoch_{curr_epoch:03d}_loss_log.csv')
                log_file = open(log_csv_path, 'w', newline='', encoding='utf-8')
                log_writer = csv.writer(log_file)
                log_writer.writerow(['epoch', 'step', 'predictor_loss', 'user_infoNCE_loss', 'echo_chamber_debiased_loss', 'total_loss', 'mean_loss', 'grad_norm', 'lr', 'step_time_s'])
                log_file.flush()

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

                    # fetch echo histories for this batch (always done when contrastive_mode == USER)
                    echo_his_title    = Variable(news_title[user_his_echo[train_user]]).to(device)
                    echo_his_abstract = Variable(news_abstract[user_his_echo[train_user]]).to(device)

                    model.train()
                    optimizer.zero_grad()

                    # model always returns 3-tuple during training:
                    # (predict_logits, proto_logits [B,2], echo_logits [B*(K-1),2])
                    predictor_logits, proto_logits, echo_logits = model(
                        candidate_title, candidate_abstract,
                        his_title, his_abstract,
                        echo_his_title, echo_his_abstract
                    )
                    predictor_loss = criterion(predictor_logits, train_label)

                    if contrastive_mode == 'USER':
                        # labels are zeros: index 0 in logits is always the positive
                        proto_labels = torch.zeros(len(proto_logits), dtype=torch.long, device=device)
                        echo_labels  = torch.zeros(len(echo_logits),  dtype=torch.long, device=device)
                        proto_loss   = F.cross_entropy(proto_logits, proto_labels)
                        echo_loss    = F.cross_entropy(echo_logits,  echo_labels)
                        loss = predictor_loss + alpha * proto_loss + beta * echo_loss
                        p_val     = predictor_loss.item()
                        proto_val = proto_loss.item()
                        echo_val  = echo_loss.item()
                        print('predictor: {:.4f}  proto_CL: {:.4f}  echo_CL: {:.4f}  total: {:.4f}'.format(
                            p_val, proto_val, echo_val, loss.item()))
                    else:
                        loss = predictor_loss
                        p_val     = predictor_loss.item()
                        proto_val = 0.0
                        echo_val  = 0.0
                        print('predictor_loss: ', predictor_loss.data.item())

                    # gradient clipping to prevent loss spikes / exploding gradients
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
                    optimizer.step()

                    step_time = time.time() - t1
                    loss_per_epoch.append(loss.data.item())
                    curr_step  = step + 1
                    curr_mean  = float(np.mean(loss_per_epoch))
                    curr_lr    = optimizer.param_groups[0]['lr']

                    # Log metrics to per-epoch CSV file in real time
                    log_writer.writerow([
                        curr_epoch, curr_step, round(p_val, 6), round(proto_val, 6),
                        round(echo_val, 6), round(loss.item(), 6), round(curr_mean, 6),
                        round(grad_norm, 6), curr_lr, round(step_time, 4)
                    ])
                    log_file.flush()

                    print('epoch: {:04d}'.format(curr_epoch),
                          'step: {:04d}'.format(curr_step),
                          'loss: {:.4f}'.format(curr_mean),
                          'time: {:.4f}'.format(step_time))

                log_file.close()
                torch.save(model.state_dict(), preserve_dir + '/model_{}.pkl'.format(curr_epoch))
                print('epoch: {:04d}'.format(curr_epoch), 'time: {:.4f}'.format(time.time() - t0))
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

    # Determine which epochs to evaluate
    epochs_to_eval = range(1, num_dataset * num_epoch + 1)
    if args.eval_epoch > 0:
        epochs_to_eval = [args.eval_epoch]

    for epoch_idx in epochs_to_eval:
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
                t1 = time.time()
                print ('index_of_batch_valdataset: ', step)

                candidate_title, his_title = news_title[val_candidate].unsqueeze(dim = 1).to(device), news_title[user_his[val_user]].to(device)
                candidate_title, his_title = Variable(candidate_title), Variable(his_title)
                candidate_abstract, his_abstract = news_abstract[val_candidate].unsqueeze(dim = 1).to(device), news_abstract[user_his[val_user]].to(device)
                candidate_abstract, his_abstract = Variable(candidate_abstract), Variable(his_abstract)
                print (candidate_title.size(), his_title.size(), candidate_abstract.size(), his_abstract.size())

                predictor_logits, _, __ = model(candidate_title, candidate_abstract, his_title, his_abstract)
                # prob of clicking on that article
                score = torch.sigmoid(predictor_logits).cpu().data.numpy()
                val_score = val_score + score.tolist()
            print('val_time: {:.4f}'.format(time.time() - t), 'val_score.length: ', len(val_score))
        f = open(preserve_dir + '/val_score_{}.pkl'.format(epoch_idx), 'wb')
        pickle.dump(val_score, f)
        f.close()

        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(epoch_idx), 'w')
        print ('process predict_file_{} start'.format(epoch_idx))

        cnt = 0
        for idx, i in enumerate(val_index):
            # extract the list of scores for all the correponding news artciles using the obtained indices
            i_score = [item for item in val_score[i[0]: i[1]]]
            # sort the scores
            i_score_sort = sorted(i_score, reverse=True)
            
            rank = []
            for item in i_score:
                # obtain the rank for the articles based on their position in the sorted score list
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
        
