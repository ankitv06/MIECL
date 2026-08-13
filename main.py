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
    # --- Training ---
    parser.add_argument('--num_epoch',   type=int,   default=1)
    parser.add_argument('--num_dataset', type=int,   default=1)
    parser.add_argument('--batch_size',  type=int,   default=30)
    parser.add_argument('--lr',          type=float, default=0.001)
    parser.add_argument('--weight_decay',type=float, default=1e-4)
    # --- Model architecture ---
    parser.add_argument('--hid_dim',      type=int,   default=400)
    parser.add_argument('--num_head',     type=int,   default=20)
    parser.add_argument('--num_prototype',type=int,   default=5)
    parser.add_argument('--word_dim',     type=int,   default=300)
    parser.add_argument('--dropout_rate', type=float, default=0.1)
    parser.add_argument('--num_negative_sample', type=int, default=3)
    parser.add_argument('--multi_rep_mode', type=str, default='concat')
    parser.add_argument('--infonce_mode',   type=str, default='prototype_self',
                        choices=['prototype_self', 'echo_chamber_debiased',
                                 'popularity_debiased', 'all_combined'])
    parser.add_argument('--contrastive_mode', type=str, default='USER')
    parser.add_argument('--gnn_mode',  type=str, default='nogat')
    parser.add_argument('--agg_mode',  type=str, default='soft')
    parser.add_argument('--pretrain_method', type=str, default='glove')
    # --- Loss weights (one per CL path) ---
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for prototype_self CL loss (always active)')
    parser.add_argument('--beta',  type=float, default=0.2,
                        help='Weight for echo_chamber_debiased CL loss')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Weight for popularity_debiased CL loss')
    # --- Temperatures (separate per CL path) ---
    parser.add_argument('--temp_proto', type=float, default=0.1,
                        help='Temperature for prototype_self InfoNCE')
    parser.add_argument('--temp_echo',  type=float, default=0.07,
                        help='Temperature for echo_chamber_debiased InfoNCE')
    parser.add_argument('--temp_pop',   type=float, default=0.2,
                        help='Temperature for popularity_debiased InfoNCE')
    # --- Paths ---
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Root dir containing MINDsmall_train/ and MINDsmall_dev/')
    parser.add_argument('--glove_path',  type=str, required=True,
                        help='Path to glove.840B.300d.txt')
    parser.add_argument('--preserve_dir', type=str, required=True,
                        help='Output directory for model checkpoints, logs, scores')
    # --- Misc ---
    parser.add_argument('--val_only', action='store_true',
                        help='Skip training and go straight to validation')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Batch size to use during evaluation to prevent OOM (default 32).')
    parser.add_argument('--eval_epochs', type=int, nargs='+', default=[],
                        help='Specific epochs to evaluate (e.g., --eval_epochs 1 20 200). If empty, evaluates all.')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume training from this epoch. Loads model_{n}.pkl and skips epochs 1..n.')
    args = parser.parse_args()

    num_epoch        = args.num_epoch
    num_dataset      = args.num_dataset
    batch_size       = args.batch_size
    lr               = args.lr
    weight_decay     = args.weight_decay
    hid_dim          = args.hid_dim
    num_head         = args.num_head
    num_prototype    = args.num_prototype
    word_dim         = args.word_dim
    dropout_rate     = args.dropout_rate
    num_negative_sample = args.num_negative_sample
    multi_rep_mode   = args.multi_rep_mode
    infonce_mode     = args.infonce_mode
    contrastive_mode = args.contrastive_mode
    gnn_mode         = args.gnn_mode
    agg_mode         = args.agg_mode
    pretrain_method  = args.pretrain_method
    alpha            = args.alpha
    beta             = args.beta
    gamma            = args.gamma
    temp_proto       = args.temp_proto
    temp_echo        = args.temp_echo
    temp_pop         = args.temp_pop
    dataset_dir      = args.dataset_dir
    glove_path       = args.glove_path
    preserve_dir     = args.preserve_dir
    val_only         = args.val_only
    eval_batch_size  = args.eval_batch_size
    resume_epoch     = args.resume_epoch

    if not os.path.exists(preserve_dir):
        os.makedirs(preserve_dir)

    file1 = os.path.join(dataset_dir, 'MINDsmall_train/news.tsv')
    file2 = os.path.join(dataset_dir, 'MINDsmall_dev/news.tsv')
    file3 = os.path.join(dataset_dir, 'MINDsmall_train/behaviors.tsv')
    file4 = os.path.join(dataset_dir, 'MINDsmall_dev/behaviors.tsv')
    file5 = glove_path
    file6 = os.path.join(dataset_dir, 'MINDsmall_dev/behaviors.tsv')  # dummy fallback

    data_module = DataProcess(file1, file2, file3, file4, file5, file6)
    news_title, news_abstract = data_module.process_train_val_news()
    news_title  = torch.LongTensor(news_title)
    news_abstract = torch.LongTensor(news_abstract)

    # Popularity pools — must be called before pre_train_behaviors()
    data_module.compute_popularity()

    entity_matrix = data_module.generate_entity_matrix()
    entity_dim = entity_matrix.size(1)

    word_matrix = None
    if pretrain_method == 'glove':
        word_matrix = data_module.load_glove()

    user_his = data_module.generate_user_his()
    user_his = torch.LongTensor(np.array(list(user_his.values()), dtype='int32'))
    print('num_user: ', len(user_his))

    # --- Echo-Chamber Debiasing: pre-compute augmented user histories ---
    # Generated when infonce_mode requires echo path.
    user_his_echo = None
    if infonce_mode in ('echo_chamber_debiased', 'all_combined'):
        print('Generating echo-chamber user histories...')
        user_his_echo_raw = data_module.generate_echo_user_his(echo_threshold=0.85)
        user_his_echo = torch.LongTensor(np.array(list(user_his_echo_raw.values()), dtype='int32'))
        print('user_his_echo.size: ', user_his_echo.size())

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device:', device)

    model = Multi_Rep_Predictor(
        num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix,
        num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode,
        gnn_mode, agg_mode,
        temp_proto=temp_proto, temp_echo=temp_echo, temp_pop=temp_pop
    )
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model = model.to(device)

    # --- Resume from checkpoint ---
    if resume_epoch > 0:
        ckpt_path = os.path.join(preserve_dir, f'model_{resume_epoch}.pkl')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Resumed from checkpoint: {ckpt_path}')
        print(f'Skipping epochs 1 to {resume_epoch}, resuming from epoch {resume_epoch + 1}.')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # CosineAnnealingLR: decays lr smoothly from lr -> 1e-5 over all epochs
    T_max = max(num_epoch * num_dataset, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)
    # Fast-forward scheduler to match the resumed epoch
    for _ in range(resume_epoch):
        scheduler.step()

    best_epoch = 0
    min_loss   = float('inf')

    if not val_only:
        for n_d in range(num_dataset):
            # pre_train_behaviors returns 6 tensors (3 base + 3 popularity triplet)
            [train_candidate, train_user, train_label,
             train_pop, train_unpop, train_diff] = data_module.pre_train_behaviors()

            train_dataset = Data.TensorDataset(
                train_candidate, train_user, train_label,
                train_pop, train_unpop, train_diff
            )
            train_loader = Data.DataLoader(
                dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
            )

            for n_ep in range(num_epoch):
                curr_epoch = n_d * num_epoch + n_ep + 1

                # Skip already-completed epochs when resuming
                if curr_epoch <= resume_epoch:
                    print(f'Skipping epoch {curr_epoch} (already trained).')
                    continue

                # Open per-epoch CSV log
                log_path   = os.path.join(preserve_dir, f'epoch_{curr_epoch:03d}_loss_log.csv')
                log_file   = open(log_path, 'w', newline='', encoding='utf-8')
                log_writer = csv.writer(log_file)
                log_writer.writerow([
                    'epoch', 'step',
                    'predictor_loss', 'proto_CL_loss',
                    'echo_CL_loss', 'pop_CL_loss',
                    'total_loss', 'mean_loss',
                    'grad_norm', 'lr', 'step_time_s'
                ])
                log_file.flush()

                t0 = time.time()
                loss_per_epoch = []

                for step, (train_candidate, train_user, train_label,
                            train_pop, train_unpop, train_diff) in enumerate(train_loader):
                    t1 = time.time()

                    # Real history
                    candidate_title   = Variable(news_title[train_candidate].to(device))
                    his_title         = Variable(news_title[user_his[train_user]].to(device))
                    candidate_abstract = Variable(news_abstract[train_candidate].to(device))
                    his_abstract      = Variable(news_abstract[user_his[train_user]].to(device))
                    train_label       = Variable(train_label.to(device))

                    # Echo history (only if echo path is active)
                    echo_his_title    = None
                    echo_his_abstract = None
                    if user_his_echo is not None:
                        echo_his_title    = Variable(news_title[user_his_echo[train_user]].to(device))
                        echo_his_abstract = Variable(news_abstract[user_his_echo[train_user]].to(device))

                    # Popularity triplet (always fetched; model ignores when mode doesn't need it)
                    pop_title    = Variable(news_title[train_pop].unsqueeze(1).to(device))
                    pop_abstract = Variable(news_abstract[train_pop].unsqueeze(1).to(device))
                    unpop_title    = Variable(news_title[train_unpop].unsqueeze(1).to(device))
                    unpop_abstract = Variable(news_abstract[train_unpop].unsqueeze(1).to(device))
                    diff_title    = Variable(news_title[train_diff].unsqueeze(1).to(device))
                    diff_abstract = Variable(news_abstract[train_diff].unsqueeze(1).to(device))

                    model.train()
                    optimizer.zero_grad()

                    predictor_logits, logits_dict = model(
                        candidate_title, candidate_abstract,
                        his_title, his_abstract,
                        echo_his_title, echo_his_abstract,
                        pop_title, pop_abstract,
                        unpop_title, unpop_abstract,
                        diff_title, diff_abstract
                    )

                    predictor_loss = criterion(predictor_logits, train_label)
                    loss           = predictor_loss
                    proto_val = echo_val = pop_val = 0.0

                    if contrastive_mode == 'USER':
                        # --- prototype_self loss (always active) ---
                        proto_labels = torch.zeros(len(logits_dict['proto']), dtype=torch.long, device=device)
                        proto_loss   = F.cross_entropy(logits_dict['proto'], proto_labels)
                        proto_val    = proto_loss.item()
                        loss         = loss + alpha * proto_loss

                        # --- echo_chamber_debiased loss ---
                        if logits_dict['echo'] is not None:
                            echo_labels = torch.zeros(len(logits_dict['echo']), dtype=torch.long, device=device)
                            echo_loss   = F.cross_entropy(logits_dict['echo'], echo_labels)
                            echo_val    = echo_loss.item()
                            loss        = loss + beta * echo_loss

                        # --- popularity_debiased loss (mask sentinel rows) ---
                        if logits_dict['pop'] is not None:
                            valid_rows = (logits_dict['pop'].abs().sum(dim=-1) > 0)
                            if valid_rows.any():
                                pop_labels = torch.zeros(valid_rows.sum(), dtype=torch.long, device=device)
                                pop_loss   = F.cross_entropy(logits_dict['pop'][valid_rows], pop_labels)
                                pop_val    = pop_loss.item()
                                loss       = loss + gamma * pop_loss

                    print('predictor: {:.4f}  proto_CL: {:.4f}  echo_CL: {:.4f}  pop_CL: {:.4f}  total: {:.4f}'.format(
                        predictor_loss.item(), proto_val, echo_val, pop_val, loss.item()))

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).item()
                    optimizer.step()

                    step_time = time.time() - t1
                    loss_per_epoch.append(loss.item())
                    curr_step = step + 1
                    curr_mean = float(np.mean(loss_per_epoch))
                    curr_lr   = optimizer.param_groups[0]['lr']

                    log_writer.writerow([
                        curr_epoch, curr_step,
                        round(predictor_loss.item(), 6), round(proto_val, 6),
                        round(echo_val, 6), round(pop_val, 6),
                        round(loss.item(), 6), round(curr_mean, 6),
                        round(grad_norm, 6), curr_lr, round(step_time, 4)
                    ])
                    log_file.flush()

                    print('epoch: {:04d}  step: {:04d}  mean_loss: {:.4f}  time: {:.4f}'.format(
                        curr_epoch, curr_step, curr_mean, step_time))

                scheduler.step()
                log_file.close()
                torch.save(model.state_dict(), os.path.join(preserve_dir, 'model_{}.pkl'.format(curr_epoch)))
                print('epoch: {:04d}  epoch_time: {:.4f}'.format(curr_epoch, time.time() - t0))

            del train_candidate, train_user, train_label, train_pop, train_unpop, train_diff
    else:
        print('Skipping training, starting validation from saved checkpoint...')

    print('TRAINING DONE' + '-' * 80)
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

    #val_candidate = np.array_split(val_candidate, 8000)     # [7600, 1800] , [11400, 1200], [22800, 600], [15200, 900]
    #val_user = np.array_split(val_user, 8000)       # [9120, 1500] , [34200, 400], [30400, 450]
    # [90, 26600] [400, 6600]

    if args.eval_epochs:
        # (c) explicit list — evaluate exactly these epochs, nothing else
        epochs_to_eval = args.eval_epochs
    elif val_only:
        # (b) --val_only — discover every model_*.pkl present in preserve_dir
        found = sorted(
            int(f.split('_')[1].split('.')[0])
            for f in os.listdir(preserve_dir)
            if f.startswith('model_') and f.endswith('.pkl')
        )
        if not found:
            raise FileNotFoundError(f'No model_*.pkl checkpoints found in {preserve_dir}')
        print(f'--val_only: found {len(found)} checkpoints -> epochs {found}')
        epochs_to_eval = found
    else:
        # (a) normal run or resume — always evaluate all epochs 1..total
        epochs_to_eval = range(1, num_dataset * num_epoch + 1)

    for epoch_idx in epochs_to_eval:
        #model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode)
        #loaded_dict = torch.load(preserve_dir + '/model_{}.pkl'.format(epoch_idx))
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

                #temp_candidate_title, temp_his_title = news_title[torch.LongTensor(val_candidate[i])].unsqueeze(dim = 1).cuda(), news_title[user_his[torch.LongTensor(val_user[i])]].cuda()
                candidate_title, his_title = news_title[val_candidate].unsqueeze(dim = 1), news_title[user_his[val_user]]
                candidate_title, his_title = Variable(candidate_title), Variable(his_title)
                #temp_candidate_abstract, temp_his_abstract = news_abstract[torch.LongTensor(val_candidate[i])].unsqueeze(dim = 1).cuda(), news_abstract[user_his[torch.LongTensor(val_user[i])]].cuda()
                candidate_abstract, his_abstract = news_abstract[val_candidate].unsqueeze(dim = 1), news_abstract[user_his[val_user]]
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
        f = open(preserve_dir + '/val_score_{}.pkl'.format(epoch_idx), 'wb')
        pickle.dump(val_score, f)
        f.close()

        #f1 = open(preserve_dir + '/val_index.pkl', 'rb')
        #f2 = open(preserve_dir + '/val_score_{}.pkl'.format(epoch_idx), 'rb')
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
        
