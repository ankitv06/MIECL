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
    parser.add_argument('--alpha1', type=float, default=1.0,
                        help='Weight for prototype contrastive loss (used in CF mode)')
    parser.add_argument('--alpha2', type=float, default=1.0,
                        help='Weight for counterfactual contrastive loss (used in CF mode)')
    parser.add_argument('--temp_proto', type=float, default=0.1,
                        help='Temperature for prototype contrastive loss')
    parser.add_argument('--temp_cf', type=float, default=0.1,
                        help='Temperature for counterfactual contrastive loss')
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
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Resume training from this epoch checkpoint. E.g., --resume_epoch 24 loads model_24.pkl and trains from epoch 25 onwards.')
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
    alpha1 = args.alpha1
    alpha2 = args.alpha2
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

    # ── Counterfactual augmentation: auto-generate cf files if needed ──────────
    cf_tsv      = os.path.join(dataset_dir, 'MINDsmall_train/news_cf.tsv')
    cf_map_json = os.path.join(dataset_dir, 'MINDsmall_train/news_cf_map.json')
    if contrastive_mode == 'CF':
        if not (os.path.exists(cf_tsv) and os.path.exists(cf_map_json)):
            print('[main] CF mode: news_cf.tsv not found — running generate_cf_news.py...')
            from generate_cf_news import run as run_cf_gen
            run_cf_gen(dataset_dir)
            if not (os.path.exists(cf_tsv) and os.path.exists(cf_map_json)):
                raise RuntimeError('generate_cf_news failed to produce output files. Aborting.')
        else:
            print('[main] CF mode: news_cf.tsv already exists — skipping generation.')
    # ──────────────────────────────────────────────────────────────────────────

    data_module = DataProcess(file1, file2, file3, file4, file5, file6)

    if contrastive_mode == 'CF':
        news_title, news_abstract, news_title_cf, news_abstract_cf = \
            data_module.process_train_val_news(cf_tsv=cf_tsv, cf_map_json=cf_map_json)
        news_title_cf    = torch.LongTensor(news_title_cf)
        news_abstract_cf = torch.LongTensor(news_abstract_cf)
    else:
        news_title, news_abstract, _, _ = data_module.process_train_val_news()
        news_title_cf    = None
        news_abstract_cf = None

    news_title, news_abstract = torch.LongTensor(news_title), torch.LongTensor(news_abstract)
    
    entity_matrix = data_module.generate_entity_matrix()
    entity_dim = entity_matrix.size(1)

    word_matrix = None
    if pretrain_method == 'glove':
        word_matrix = data_module.load_glove()

    if contrastive_mode == 'CF':
        user_his, user_his_cf = data_module.generate_user_his()
        user_his_cf = torch.LongTensor(np.array(list(user_his_cf.values()), dtype='int32'))
    else:
        user_his, _ = data_module.generate_user_his()
        user_his_cf = None

    user_his = torch.LongTensor(np.array(list(user_his.values()), dtype = 'int32'))
    print ('num_user: ', len(user_his))
    
    #user_adj = []
    #f = open('small_user_nei_sort.txt', 'r', encoding='utf-8')
    #lines = f.readlines()
    #for line in lines:
    #    line = line.strip().split('\t')
    #    user_adj.append([int(i) for i in line])
    #user_adj = torch.LongTensor(np.array(user_adj, dtype = 'int32'))
    #print ('user_adj.size: ', user_adj.size())

    temp_proto = args.temp_proto
    temp_cf = args.temp_cf

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Multi_Rep_Predictor(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode, temp_proto, temp_cf)
    model = model.to(device)
    if torch.cuda.is_available():
        model = nn.DataParallel(model)
    
    #model.load_state_dict(torch.load('...'))
    #model.load_state_dict(torch.load('...'))
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    #optimizer = optim.Adamax(model.parameters(), lr=0.002)
    
    best_epoch = 0
    min_loss = float('inf')

    if not args.eval_only:
        if args.resume_epoch > 0:
            resume_path = os.path.join(preserve_dir, 'model_{}.pkl'.format(args.resume_epoch))
            if not os.path.exists(resume_path):
                raise FileNotFoundError('Resume checkpoint not found: {}'.format(resume_path))
            print('Resuming training from epoch {} (loaded {})'.format(args.resume_epoch + 1, resume_path))
            model.load_state_dict(torch.load(resume_path, map_location=device))

        for n_d in range(num_dataset):
            # training loop
            [train_candidate, train_user, train_label] = data_module.pre_train_behaviors()
            train_dataset = Data.TensorDataset(train_candidate, train_user, train_label)
            train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
            
            for n_ep in range(num_epoch):
                current_epoch = n_d * num_epoch + n_ep + 1
                if args.resume_epoch > 0 and current_epoch <= args.resume_epoch:
                    continue
                acc, all = 0, 0
                t0 = time.time()
                loss_per_epoch = []

                # Initialize CSV logging for this epoch
                epoch_csv_path = os.path.join(preserve_dir, f'epoch_{current_epoch:03d}_loss_log.csv')
                epoch_csv_file = open(epoch_csv_path, 'w', newline='')
                csv_writer = csv.writer(epoch_csv_file)
                csv_writer.writerow(['epoch', 'step', 'predictor_loss', 'proto_CL_loss', 'cf_loss', 'total_loss', 'mean_loss', 'grad_norm', 'lr', 'step_time_s'])

                # batches from the training loader
                for step, (train_candidate, train_user, train_label) in enumerate(train_loader):
                    t1 = time.time()
                    candidate_title, his_title, train_label = news_title[train_candidate].to(device), news_title[user_his[train_user]].to(device), train_label.to(device)
                    candidate_title, his_title, train_label = Variable(candidate_title),Variable(his_title), Variable(train_label)
                    candidate_abstract, his_abstract  = news_abstract[train_candidate].to(device), news_abstract[user_his[train_user]].to(device)
                    candidate_abstract, his_abstract  = Variable(candidate_abstract),Variable(his_abstract)
                    print (candidate_title.size(), candidate_abstract.size())

                    # Load CF history if in CF mode
                    if contrastive_mode == 'CF':
                        his_title_cf    = news_title_cf[user_his_cf[train_user]].to(device)
                        his_abstract_cf = news_abstract_cf[user_his_cf[train_user]].to(device)
                        his_title_cf    = Variable(his_title_cf)
                        his_abstract_cf = Variable(his_abstract_cf)
                    else:
                        his_title_cf    = None
                        his_abstract_cf = None
    
                    model.train()
                    optimizer.zero_grad()
    
                    predictor_logits, proto_logits, cf_logits = model(
                        candidate_title, candidate_abstract,
                        his_title, his_abstract,
                        his_title_cf, his_abstract_cf
                    )
                    predictor_loss = criterion(predictor_logits, train_label)
                    
                    val_pred = predictor_loss.item()
                    val_proto = 0.0
                    val_cf = 0.0

                    if contrastive_mode == 'USER':
                        user_infoNCE_labels = torch.zeros(len(proto_logits), dtype=torch.long, device=device)
                        user_infoNCE_loss = F.cross_entropy(proto_logits, user_infoNCE_labels)
                        val_proto = user_infoNCE_loss.item()
                        print ('predictor_loss: ', predictor_loss.data.item(), 'user_infoNCE_loss: ', user_infoNCE_loss.data.item())
                        loss = predictor_loss + alpha * user_infoNCE_loss

                    elif contrastive_mode == 'CF':
                        proto_labels = torch.zeros(len(proto_logits), dtype=torch.long, device=device)
                        cf_labels    = torch.zeros(len(cf_logits),    dtype=torch.long, device=device)
                        proto_loss   = F.cross_entropy(proto_logits, proto_labels)
                        cf_loss      = F.cross_entropy(cf_logits,    cf_labels)
                        val_proto    = proto_loss.item()
                        val_cf       = cf_loss.item()
                        print('predictor_loss:', predictor_loss.data.item(),
                              'proto_loss:', proto_loss.data.item(),
                              'cf_loss:', cf_loss.data.item())
                        loss = predictor_loss + alpha1 * proto_loss + alpha2 * cf_loss

                    else:
                        print ('predictor_loss: ', predictor_loss.data.item())
                        loss = predictor_loss
                        
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    if isinstance(grad_norm, torch.Tensor):
                        grad_norm_val = grad_norm.item()
                    else:
                        grad_norm_val = grad_norm
                        
                    optimizer.step()
    
                    loss_per_epoch.append(loss.data.item())
                    step_time = time.time() - t1
                    
                    csv_writer.writerow([
                        current_epoch, step + 1,
                        val_pred, val_proto, val_cf,
                        loss.item(), np.mean(loss_per_epoch),
                        grad_norm_val, optimizer.param_groups[0]['lr'],
                        step_time
                    ])
                    epoch_csv_file.flush()
                    
                    print('epoch: {:04d}'.format(n_d * num_epoch + n_ep + 1), 'step: {:04d}'.format(step + 1), 'loss: {:.4f}'.format(np.mean(loss_per_epoch)), 'time: {:.4f}'.format(step_time))

                epoch_csv_file.close()
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

    if args.eval_epochs:
        epochs_to_eval = args.eval_epochs
    elif args.eval_only:
        existing_pkls = glob.glob(os.path.join(preserve_dir, 'model_*.pkl'))
        epochs_to_eval = sorted([
            int(os.path.basename(p).replace('model_', '').replace('.pkl', ''))
            for p in existing_pkls
        ])
        if not epochs_to_eval:
            print('No model_*.pkl files found in {}'.format(preserve_dir))
        else:
            print('eval_only: found checkpoints for epochs: {}'.format(epochs_to_eval))
    else:
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
            
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
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

                model_out = model(candidate_title, candidate_abstract, his_title, his_abstract)
                predictor_logits = model_out[0]
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
