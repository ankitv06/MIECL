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
    parser.add_argument('--num_negative_sample', type=int, default=3)
    parser.add_argument('--word_dim', type=int, default=300)
    parser.add_argument('--preserve_dir', type=str, default='dataset/outputs-after-200-epochs')
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

    # Compute article popularity and build per-topic pools for augmentation
    # Must be called after process_train_val_news() (needs news_id and category maps)
    # and before pre_train_behaviors() (which uses the pools)
    data_module.compute_popularity()
    
    entity_matrix = data_module.generate_entity_matrix()
    entity_dim = entity_matrix.size(1)

    word_matrix = None
    if pretrain_method == 'glove':
        word_matrix = data_module.load_glove()

    user_his = data_module.generate_user_his()
    user_his = torch.LongTensor(np.array(list(user_his.values()), dtype = 'int32'))
    print ('num_user: ', len(user_his))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    model = model.to(device)
 
    #user_adj = []
    #f = open('small_user_nei_sort.txt', 'r', encoding='utf-8')
    #lines = f.readlines()
    #for line in lines:
    #    line = line.strip().split('\t')
    #    user_adj.append([int(i) for i in line])
    #user_adj = torch.LongTensor(np.array(user_adj, dtype = 'int32'))
    #print ('user_adj.size: ', user_adj.size())
    


    #model.load_state_dict(torch.load('/home/wangshicheng/news_recommendation/Final_edtion/title_abstract_edition/concat_dr0.0_prototype_other_user_nogat_soft_6_3_5_s/model_{}.pkl'.format(i + 1)))
    #model.load_state_dict(torch.load('/home/wangshicheng/news_recommendation/Final_edtion/title_abstract_edition/sgd_other2_10_1_5_l_adam_val_2/model_6.pkl'))
    model = model
    
    criterion = nn.BCEWithLogitsLoss()
    # Use lr and weight_decay from argparse (defaults: lr=0.001, weight_decay=1e-4).
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    #optimizer = optim.Adamax(model.parameters(), lr=0.002)

    # FIX 5: CosineAnnealingLR scheduler.
    # Smoothly decays lr from `lr` (e.g. 0.001) down toward eta_min=1e-5 over
    # T_max=num_epoch steps. This means:
    #   - Early epochs: lr is high -> fast convergence
    #   - Late epochs:  lr is low  -> fine-tuned updates, harder to diverge
    #   - If a loss spike happens late in training, the smaller lr limits recovery
    #     overshoot. Without a scheduler, lr=0.001 stays constant for 200 epochs.
    T_max = max(num_epoch * num_dataset, 1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-5)

    best_epoch = 0
    min_loss = float('inf')

    for n_d in range(num_dataset):
        if args.eval_only:
            print("Skipping training, starting validation from saved checkpoint...")
            break
        # training loop — pre_train_behaviors now returns 6 tensors
        [train_candidate, train_user, train_label,
         train_pop, train_unpop, train_diff] = data_module.pre_train_behaviors()
        train_dataset = Data.TensorDataset(train_candidate, train_user, train_label,
                                           train_pop, train_unpop, train_diff)
        train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        for n_ep in range(num_epoch):
            acc, all = 0, 0
            t0 = time.time()
            loss_per_epoch = []

            # Open a fresh CSV log file for this epoch.
            # Each epoch gets its own file: epoch_0001_loss_log.csv, epoch_0002_loss_log.csv, …
            # Columns logged every step:
            #   epoch          – global epoch number (n_d * num_epoch + n_ep + 1)
            #   step           – batch index within the epoch (1-indexed)
            #   predictor_loss – BCE between predicted scores and click labels (main task)
            #   user_infoNCE   – contrastive loss aligning user rep with interest prototype
            #   news_debiased  – contrastive loss pushing same-topic reps together (our addition)
            #   total_loss     – weighted sum: pred + 0.1*NCE + 0.1*debiased
            #   mean_loss      – running mean of total_loss over steps seen so far this epoch
            #   grad_norm      – L2 norm of gradients BEFORE clipping (>1.0 means clip fired)
            #   lr             – current learning rate from the scheduler
            #   step_time_s    – wall-clock seconds for this single step
            global_epoch = n_d * num_epoch + n_ep + 1
            log_path = os.path.join(preserve_dir, 'epoch_{:04d}_loss_log.csv'.format(global_epoch))
            log_file = open(log_path, 'w', newline='', encoding='utf-8')
            log_writer = csv.writer(log_file)
            log_writer.writerow([
                'epoch', 'step',
                'predictor_loss', 'user_infoNCE_loss', 'news_debiased_loss',
                'total_loss', 'mean_loss',
                'grad_norm', 'lr', 'step_time_s'
            ])

            # batches from the training loader
            # news titles and abstracts are obtained based on user behavior
            # neighboring users and corresponding news titles and abstracts are obtained
            # model set to training mode + gradients set to 0
            # model saved after every epoch
            for step, (train_candidate, train_user, train_label,
                        train_pop, train_unpop, train_diff) in enumerate(train_loader):
                t1 = time.time()
                candidate_title, his_title = news_title[train_candidate].to(device), news_title[user_his[train_user]].to(device)
                train_label = train_label.to(device)
                candidate_title, his_title, train_label = Variable(candidate_title), Variable(his_title), Variable(train_label)
                candidate_abstract, his_abstract  = news_abstract[train_candidate].to(device), news_abstract[user_his[train_user]].to(device)
                candidate_abstract, his_abstract  = Variable(candidate_abstract), Variable(his_abstract)
                print (candidate_title.size(), candidate_abstract.size())

                # --- Popularity Debiased Augmentation: look up titles/abstracts for triplet ---
                # Each of train_pop/unpop/diff is [batch] of news int_ids.
                # news_title shape: [num_news, 20]. Lookup gives [batch, 20]; unsqueeze to [batch, 1, 20].
                pop_title    = Variable(news_title[train_pop].unsqueeze(1).to(device))
                pop_abstract = Variable(news_abstract[train_pop].unsqueeze(1).to(device))
                unpop_title    = Variable(news_title[train_unpop].unsqueeze(1).to(device))
                unpop_abstract = Variable(news_abstract[train_unpop].unsqueeze(1).to(device))
                diff_title    = Variable(news_title[train_diff].unsqueeze(1).to(device))
                diff_abstract = Variable(news_abstract[train_diff].unsqueeze(1).to(device))

                model.train()
                optimizer.zero_grad()

                predictor_logits, user_infoNCE_logits, news_debiased_logits = model(
                    candidate_title, candidate_abstract, his_title, his_abstract,
                    pop_title, pop_abstract, unpop_title, unpop_abstract, diff_title, diff_abstract)
                predictor_loss = criterion(predictor_logits, train_label)
                
                if contrastive_mode == 'USER':
                    user_infoNCE_labels = torch.zeros(len(user_infoNCE_logits), dtype=torch.long).to(device)
                    user_infoNCE_loss = F.cross_entropy(user_infoNCE_logits, user_infoNCE_labels)

                    # --- Popularity Debiased CL loss (beta = 0.5) ---
                    # Only include non-sentinel rows (those not zeroed by the mask in model.py).
                    # A zeroed row has both logits == 0; we exclude it to avoid pulling the loss
                    # toward a trivially-satisfied example.
                    if news_debiased_logits is not None:
                        valid_rows = (news_debiased_logits.abs().sum(dim=-1) > 0)  # [batch] bool
                        if valid_rows.any():
                            debiased_labels = torch.zeros(valid_rows.sum(), dtype=torch.long).to(device)
                            news_debiased_loss = F.cross_entropy(
                                news_debiased_logits[valid_rows], debiased_labels)
                        else:
                            news_debiased_loss = torch.tensor(0.0).to(device)
                    else:
                        news_debiased_loss = torch.tensor(0.0).to(device)

                    print ('predictor_loss: ', predictor_loss.data.item(),
                           'user_infoNCE_loss: ', user_infoNCE_loss.data.item(),
                           'news_debiased_loss: ', news_debiased_loss.data.item())

                    # FIX 6: Reduce auxiliary loss weights so predictor_loss dominates.
                    #
                    # Before: loss = predictor_loss + 1.0*user_infoNCE + 0.5*news_debiased
                    #   -> aux losses had equal or greater weight than the main task.
                    #   -> When aux losses exploded (e.g. user_infoNCE=4.7e11), the
                    #      gradient from the main task was completely drowned out.
                    #
                    # After:  loss = predictor_loss + 0.1*user_infoNCE + 0.1*news_debiased
                    #   -> Both aux losses are light regularizers (10% weight each).
                    #   -> predictor_loss always drives training; aux losses shape
                    #      the embedding space without hijacking the optimizer.
                    #   -> To measure debiasing contribution: run ablation with
                    #      news_debiased_weight=0.0 and compare AUC/nDCG@10.
                    loss = predictor_loss + 0.1 * user_infoNCE_loss + 0.1 * news_debiased_loss
                else:
                    print ('predictor_loss: ', predictor_loss.data.item())
                    loss = predictor_loss
                    
                loss.backward()

                # FIX 4: Gradient clipping — hard ceiling on the gradient norm.
                # After loss.backward(), PyTorch has computed all parameter gradients.
                # clip_grad_norm_ rescales the gradient vector so its L2 norm never
                # exceeds max_norm=1.0 before the optimizer takes its step.
                #
                # Why max_norm=1.0 (not 2.0):
                #   - 2.0 still allows the gradient to move parameters 2x further than
                #     a typical safe step, which is too lenient given the instability seen.
                #   - 1.0 provides a tighter bound while still allowing meaningful updates.
                #   - If clipping fires frequently (logged below), it's a sign that the
                #     normalized logits are still too large -> reduce temperature further.
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                loss_per_epoch.append(loss.data.item())
                current_lr_step = scheduler.get_last_lr()[0]
                print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'step: {:04d}'.format(step + 1), 'loss: {:.4f}'.format(np.mean(loss_per_epoch)), 'grad_norm: {:.4f}'.format(grad_norm.item()), 'time: {:.4f}'.format(time.time() - t1))

                # --- Write one row to this epoch's CSV log ---
                # All individual loss values are recorded even if contrastive_mode != 'USER'
                # (they will be logged as 0.0 in that case so the CSV schema stays consistent).
                _pred  = predictor_loss.data.item()
                _nce   = user_infoNCE_loss.data.item()   if contrastive_mode == 'USER' else 0.0
                _deb   = news_debiased_loss.data.item()  if contrastive_mode == 'USER' else 0.0
                _total = loss.data.item()
                _mean  = float(np.mean(loss_per_epoch))
                _gnorm = float(grad_norm.item())
                _lr    = current_lr_step
                _time  = float(time.time() - t1)
                log_writer.writerow([
                    n_d * num_epoch + n_ep + 1,  # epoch
                    step + 1,                    # step (1-indexed)
                    round(_pred,  6),
                    round(_nce,   6),
                    round(_deb,   6),
                    round(_total, 6),
                    round(_mean,  6),
                    round(_gnorm, 6),
                    round(_lr,    8),
                    round(_time,  4),
                ])
                # Flush after every row so the file is readable even if training crashes
                # mid-epoch — you won't lose the completed steps.
                log_file.flush()

            # FIX 5 (continued): Step the scheduler once per epoch (not per batch).
            # After each full epoch, the scheduler lowers lr slightly along the cosine curve.
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'time: {:.4f}'.format(time.time() - t0), 'lr: {:.6f}'.format(current_lr))
            torch.save(model.state_dict(), preserve_dir + '/model_{}.pkl'.format(n_d * num_epoch + n_ep + 1))

            # Close the epoch's CSV log now that all steps are written.
            log_file.close()
            print('Loss log saved: {}'.format(log_path))

        del train_candidate, train_user, train_label

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
    print(len(val_index))
    # Use enumerate so we get the impression index (0, 1, 2...) directly.
    # The original val_index.index(i) searched the full list on every iteration
    # — O(n²) for 70k impressions. enumerate is O(1) per step.
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

    epochs_to_eval = range(1, num_dataset * num_epoch + 1)
    if args.eval_epoch > 0:
        epochs_to_eval = [args.eval_epoch]

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
                candidate_title, his_title = news_title[val_candidate].unsqueeze(dim = 1).to(device), news_title[user_his[val_user]].to(device)
                candidate_title, his_title = Variable(candidate_title), Variable(his_title)
                #temp_candidate_abstract, temp_his_abstract = news_abstract[torch.LongTensor(val_candidate[i])].unsqueeze(dim = 1).cuda(), news_abstract[user_his[torch.LongTensor(val_user[i])]].cuda()
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

                predictor_logits, _, _ = model(candidate_title, candidate_abstract, his_title, his_abstract)
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

        predict_file = open(preserve_dir + '/prediction_{}.txt'.format(epoch_idx), 'w')
        print ('process predict_file_{} start'.format(epoch_idx))

        cnt = 0
        # Use enumerate for the same O(1) index reason as truth.txt above.
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
        
