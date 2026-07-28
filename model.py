from re import S
import re
import torch
from torch.functional import tensordot
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class ScaledAttention(nn.Module):
	def __init__(self, temperature, dropout_rate):
		super().__init__()
		# scaling factor applied to dot product before applying the softmax function
		self.temperature = temperature
		# dropout rate applied to attention scores
		self.dropout = nn.Dropout(dropout_rate)

	def forward(self, q, k, v):
		# scaled dor-product attention scores using matrix mul of q with tranpose of k
		# attentions scores scaled by dividng them by temperature
		# softmax function is applied along the last dimension (-1) to obtain normalized attention weights.
		# Dropout is applied to the attention weights to introduce regularization and reduce overfitting.
		# final output is computed as the weighted sum of values (v) using the obtained attention scores.
		score = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
		score = F.softmax(score, dim=-1)
		score = self.dropout(score)
		output = torch.matmul(score, v)
		return score, output

# implements multi-head attention by linearly transforming the input query, key, and value tensors and applying scaled dot-product attention in parallel across multiple heads.
# outputs from different heads are concatenated, linearly transformed again (fc), and passed through dropout to produce the final multi-head attention output.
class MultiHeadAttention(nn.Module):
	def __init__(self, num_head, embedding_dim, hid_dim, dropout_rate):
		super().__init__()
		self.num_head = num_head
		self.size_per_head = hid_dim // num_head
		self.hid_dim = hid_dim

		self.q_linear = nn.Linear(embedding_dim, hid_dim)
		self.k_linear = nn.Linear(embedding_dim, hid_dim)
		self.v_linear = nn.Linear(embedding_dim, hid_dim)
		self.fc = nn.Linear(hid_dim, hid_dim)
		self.dropout = nn.Dropout(dropout_rate)

		self.attention = ScaledAttention(temperature = self.size_per_head ** 0.5, dropout_rate = dropout_rate)
	
	def forward(self, q, k, v):  # [30, 10, 50, 400]
		sample_size = q.size()[0]
		#(sample_size)
		
		batch_size = q.size()[1]
		q_len, k_len, v_len = q.size()[2], k.size()[2], v.size()[2]


		q = self.q_linear(q).view(sample_size, batch_size, q_len, self.num_head, self.size_per_head)
		k = self.k_linear(k).view(sample_size, batch_size, k_len, self.num_head, self.size_per_head)
		v = self.v_linear(v).view(sample_size, batch_size, v_len, self.num_head, self.size_per_head)

		q, k, v = q.transpose(2, 3), k.transpose(2, 3), v.transpose(2, 3)
		score, output = self.attention(q, k, v)

		output = output.transpose(2, 3).contiguous().view(sample_size, batch_size, v_len, self.hid_dim)
		output = self.fc(output)
		output = self.dropout(output)
		return output

# used for encoding news articles using MHSA and additional addition mechanisms capturing word & entity info
# input - title and abstract of news articles
# utilizies mHSA for word and entity embeddings
# word embeddings obtained from word matrix - from glove - embedding layer (n.Embedding) + MHSA (word_attention)
# similariy for entity embeddings, entity_matrix and entity_attention
# attention weights are found for title and abstract separately
# title and abstract representations are concatenated
# processed further through linear transformations and attention mechanisms
# final repr = news_rep for each article
# final output is a tensor containing the encoded representations of news articles, reshaped to the original size of the input title tensor.

class News_Encoder(nn.Module):
	def __init__(self, num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, dropout_rate):
		super().__init__()
		self.word_embedding = nn.Embedding.from_pretrained(word_matrix, freeze = True)
		self.entity_embedding = nn.Embedding.from_pretrained(entity_matrix, freeze = False)
		self.word_attention = MultiHeadAttention(num_head, word_dim, hid_dim, dropout_rate)
		self.entity_attention = MultiHeadAttention(num_head, entity_dim, hid_dim, dropout_rate)
		self.hid_dim = hid_dim
		
		self.W1 = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj1 = nn.Parameter(torch.Tensor(200, 1))
		self.W2 = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj2 = nn.Parameter(torch.Tensor(200, 1))
		self.W_agg = nn.Parameter(torch.Tensor(hid_dim, 1))
		nn.init.xavier_uniform_(self.W1.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj1.data, gain=1.414)
		nn.init.xavier_uniform_(self.W2.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj2.data, gain=1.414)
		nn.init.xavier_uniform_(self.W_agg.data, gain=1.414)
		
		self.dropout = nn.Dropout(dropout_rate)

		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w3 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w1.data, gain = 1.414)
		nn.init.xavier_uniform_(self.w2.data, gain = 1.414)
		nn.init.xavier_uniform_(self.w3.data, gain = 1.414)

	def forward(self, title, abstract):
		title_size = title.size() # [30, 5, 30]
		title = self.word_embedding(title)
		title = self.word_attention(title, title, title)
		title_att = torch.tanh(torch.matmul(title, self.W1))   # [30, 5, 30, 400]
		title_att = torch.matmul(title_att, self.proj1)
		title_att = F.softmax(title_att, dim = 2)
		title = torch.matmul(title_att.transpose(-2, -1), title).squeeze(dim = 2)	 # [30, 5, 400]

		abstract = self.word_embedding(abstract)
		abstract = self.word_attention(abstract, abstract, abstract)
		abstract_att = torch.tanh(torch.matmul(abstract, self.W2))
		abstract_att = torch.matmul(abstract_att, self.proj2)
		abstract_att = F.softmax(abstract_att, dim = 2)
		abstract = torch.matmul(abstract_att.transpose(-2, -1), abstract).squeeze(dim = 2)
	   
		#news_rep = torch.matmul(title, self.w1) + torch.matmul(abstract, self.w2)
		
		news_rep = torch.cat((title.reshape(-1, self.hid_dim).unsqueeze(dim = 1), abstract.reshape(-1, self.hid_dim).unsqueeze(dim = 1)), dim = 1)	  # [150, 2, 400]
		att = torch.tanh(torch.matmul(news_rep, self.W_agg))
		att = F.softmax(att, dim = 1)
		news_rep = torch.matmul(att.transpose(-1, -2), news_rep).squeeze(dim = 1).reshape(title_size[0], title_size[1], self.hid_dim)

			
		return news_rep.reshape(title_size[0], title_size[1], -1)

# takes news_rep as input - encoded repr of news articles
# FORWARD
# applies diff operations based on mode : multi_rep_mode 
	# concat - concatenates the news representations with prototypes and applies linear transformations
	# att - applies attention mechanism between news repr and prototypes
	# cat - concatenates prototypes with news repr and applies linear transformations
	# trans - applies linear transformations to news repr using learnable weights
	# single - applies dropout and tanh activation to each element in news repr
	# cln_cat - combines concat and weighted combination of prototypes based on attention weights
	# cln - applies element-wise multiplication of news repr with prototypes
	# other cases - applies a weighted combination of prottypes and news reppr based on learned weights

# module is used to experiment with the diff strategies for combining or trasnforming news repr based on prototypes and learnable parameters
# the choice of multi_rep_mode determines the specific operation applied during fwd pass
class Multi_Rep_Encoder(nn.Module):
	def __init__(self, hid_dim, num_prototype, multi_rep_mode, dropout_rate):
		super().__init__()
		self.hid_dim = hid_dim
		self.num_prototype = num_prototype
		self.mode = multi_rep_mode
		self.dropout = nn.Dropout(dropout_rate)

		self.prototype = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.W = nn.Parameter(torch.Tensor(num_prototype, hid_dim, hid_dim))
		self.w = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(200, 1))

		nn.init.xavier_uniform_(self.prototype, gain=1.414)
		nn.init.xavier_uniform_(self.w1.data, gain=1.414)
		nn.init.xavier_uniform_(self.w2.data, gain=1.414)
		nn.init.xavier_uniform_(self.W.data, gain = 1.414)
		nn.init.xavier_uniform_(self.w.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj.data, gain = 1.414)

	def forward(self, news_rep):
		news_rep_size = news_rep.size()    # [30, 5, 400]
		news_rep = news_rep.reshape(-1, news_rep_size[-1])

		if self.mode == 'concat':
			news_rep = news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1)
			news_rep = torch.matmul(news_rep, self.w1) + torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep_size[0] * news_rep_size[1], 1, 1), self.w2)
			#news_rep = news_rep + torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep_size[0] * news_rep_size[1], 1, 1), self.w2)
			news_rep = self.dropout(news_rep)
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'att':
			news_rep = news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1)   # [150, 10, 400]
			prototype = self.prototype.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1)
			news_rep = torch.cat((news_rep.unsqueeze(dim = 2), prototype.unsqueeze(dim = 2)), dim = 2)
			att = torch.tanh(torch.matmul(news_rep, self.w))
			att = torch.matmul(att, self.proj)
			att = F.softmax(att, dim = 2)	  # 
			news_rep = torch.matmul(att.transpose(-1, -2), news_rep).squeeze(dim = 2)
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'cat':
			news_rep = news_rep.unsqueeze(dim = 0).repeat(self.num_prototype, 1, 1)
			prototype = self.prototype.unsqueeze(dim = 1).repeat(1, news_rep.size(1), 1)
			news_rep = torch.matmul(news_rep, self.w1) + torch.matmul(prototype, self.w2)
			news_rep = news_rep.reshape(self.num_prototype, news_rep_size[0], news_rep_size[1], -1)
			return news_rep
		
		elif self.mode == 'trans':
			news_rep = news_rep.unsqueeze(dim = 0).repeat(self.num_prototype, 1, 1) # [10, 150, 400], [10, 400, 400]
			news_rep = torch.matmul(news_rep, self.W) # [10, 150, 400]
			news_rep = news_rep.transpose(0, 1).reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, self.hid_dim) #[30, 5, 10, 400]
		
		elif self.mode == 'single':
			news_rep = self.dropout(torch.tanh(news_rep))

		elif self.mode == 'cln_cat':
			multi_news_rep = news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1)
			multi_news_rep = torch.matmul(multi_news_rep, self.w1) + torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep_size[0] * news_rep_size[1], 1, 1), self.w2)
			weight = F.softmax(torch.matmul(news_rep, self.prototype.transpose(-1, -2)), dim = -1)	  # [150, 10]
			news_rep = torch.mul(multi_news_rep, weight.unsqueeze(dim = 2).repeat(1, 1, news_rep.size(-1)))
			news_rep = self.dropout(torch.tanh(news_rep))
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'cln_1':
			weight = torch.tanh(torch.matmul(self.prototype, self.w1))	  # [5, 400]
			news_rep = torch.mul(weight.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1), news_rep.unsqueeze(dim = 1))
			#news_rep = news_rep + torch.matmul(self.prototype, self.w2).unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1)
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

		elif self.mode == 'cln':
			weight = torch.matmul(self.prototype, self.w1)	  # [5, 400]
			news_rep = torch.mul(weight.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1), news_rep.unsqueeze(dim = 1))
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep
		
		else:
			weight = torch.sigmoid(torch.matmul(self.prototype.unsqueeze(dim = 0).repeat(news_rep.size(0), 1, 1), self.w1) + torch.matmul(news_rep.unsqueeze(dim = 1).repeat(1, self.num_prototype, 1), self.w2))
			news_rep = torch.mul(weight, news_rep.unsqueeze(dim = 1))
			news_rep = news_rep.reshape(news_rep_size[0], news_rep_size[1], self.num_prototype, -1)
			return news_rep

# i/p - history rep - repr of historical news articles for multiple users
# forward method makes the following operations
	# tranposes his_rep to make shape compatible with the expected input shape for attention
	# applies a linear transformation and a projection to history_rep to obtain attention weights (att)
	# applies softmax along the third dimension of the attention weights to obtain a prob distr
	# performs weighted sum of his_rep using att weights to obtain a multi-user repr

# module is used to encode historical news repr for multiple users using att mech
# allows model to focus on relevant information in user history
class Multi_Rep_User_Encoder(nn.Module):
	def __init__(self, num_head, hid_dim, dropout_rate, num_prototype):
		super().__init__()
		self.attention = MultiHeadAttention(num_head, hid_dim, hid_dim, dropout_rate)
		#self.attention = ScaledAttention(temperature = 1.0)
		self.W = nn.Parameter(torch.Tensor(num_prototype, hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(num_prototype, 200, 1))
		nn.init.xavier_uniform_(self.W.data, gain=1.414)
		nn.init.xavier_uniform_(self.proj.data, gain=1.414)
		self.hid_dim = hid_dim
		
		self.dropout = nn.Dropout(dropout_rate)
		

	def forward(self, history_rep):   
		history_rep = history_rep.transpose(1, 2)	 
		#history_rep = self.attention(history_rep, history_rep, history_rep)
		#history_rep_size = history_rep.size()
		#history_rep = history_rep.reshape(history_rep_size[0], -1, self.hid_dim)

		att = torch.tanh(torch.matmul(history_rep, self.W))
		att = torch.matmul(att, self.proj)
		#att = att.reshape(history_rep_size[0], history_rep_size[1], history_rep_size[2], -1)
		att = F.softmax(att, dim = 2)
		
		#multi_user_rep = self.dropout(torch.matmul(att.transpose(-2, -1), history_rep.reshape(history_rep_size[0], history_rep_size[1], history_rep_size[2], -1)).squeeze(dim = 2))
		#return multi_user_rep.transpose(0, 1)
		multi_user_rep = torch.matmul(att.transpose(-2, -1), history_rep).squeeze(dim = 2)
		
		return multi_user_rep

class InfoNCE(nn.Module):
	def __init__(self, hid_dim, infonce_mode, prototype, temperature=0.07):
		super().__init__()
		self.mode = infonce_mode
		self.prototype = prototype
		self.temperature = temperature  # controls sharpness of contrastive distribution

		self.W = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.W, gain = 1.414)

	# --- Shared helper: echo_chamber_debiased logits with L2 norm + temperature ---
	# anchor  : u_k        [B, K, D]  (real user representations)
	# positive: I_k        [K, D]     (global interest prototypes)
	# negative: u_echo_k   [B, K, D]  (echo-chamber user representations)
	# dominant_indices: [B]  — dominant prototype index per user (excluded as anchor)
	# Returns [B*(K-1), 2] logits (index 0 = positive, index 1 = negative)
	def _echo_logits(self, multi_rep, echo_rep, dominant_indices):
		num_proto = multi_rep.size(1)

		# L2-normalize all representations to unit sphere before dot product
		multi_n = F.normalize(multi_rep, dim=-1)           # [B, K, D]
		proto_n = F.normalize(self.prototype, dim=-1)      # [K, D]
		echo_n  = F.normalize(echo_rep, dim=-1)            # [B, K, D]

		# cosine similarity / temperature for each (user, prototype) pair
		pos_logits = (multi_n * proto_n.unsqueeze(0)).sum(dim=-1) / self.temperature  # [B, K]
		neg_logits = (multi_n * echo_n).sum(dim=-1) / self.temperature                # [B, K]

		# stack into [B, K, 2] (index 0 = positive I_k, index 1 = negative u_echo_k)
		all_logits = torch.stack([pos_logits, neg_logits], dim=-1)  # [B, K, 2]

		# mask out dominant interest per user — only non-dominant k are used as anchor
		prototype_range   = torch.arange(num_proto, device=multi_rep.device)  # [K]
		dom_expanded      = dominant_indices.unsqueeze(1)                      # [B, 1]
		non_dominant_mask = (prototype_range.unsqueeze(0) != dom_expanded)     # [B, K]

		return all_logits[non_dominant_mask]  # [B*(K-1), 2]
	# --- End helper ---

	def forward(self, multi_rep, echo_rep, dominant_indices):
		# --- Prototype-self logits (original MIECL contrastive loss) ---
		# anchor: u_k (one interest of real user), positive: I_k (global prototype),
		# negative: u_k' (different interest of same user). k chosen randomly per batch.
		# 正负样本1:1，正样本为对应兴趣原型向量，负样本为随机抽取某兴趣条件下新闻语义表示
		positive_index = torch.randint(low=0, high=multi_rep.size(1), size=(1,))
		negative_index = torch.randint(low=0, high=multi_rep.size(1), size=(1,))
		while positive_index == negative_index:
			negative_index = torch.randint(low=0, high=multi_rep.size(1), size=(1,))

		anchor   = torch.index_select(multi_rep, dim=1, index=positive_index).squeeze(dim=1)  # [B, D]
		positive = torch.index_select(self.prototype, dim=0, index=positive_index)             # [1, D]
		negative = torch.index_select(multi_rep, dim=1, index=negative_index).squeeze(dim=1)  # [B, D]

		# L2 normalize to unit sphere — prevents dot products from growing unbounded
		anchor_n   = F.normalize(anchor, dim=-1)
		positive_n = F.normalize(positive, dim=-1)
		negative_n = F.normalize(negative, dim=-1)

		# cosine similarity scaled by temperature
		pos_logit    = torch.matmul(anchor_n, positive_n.t()) / self.temperature            # [B, 1]
		neg_logit    = (anchor_n * negative_n).sum(dim=-1, keepdim=True) / self.temperature # [B, 1]
		proto_logits = torch.cat([pos_logit, neg_logit], dim=-1)                            # [B, 2]

		# --- Echo-Chamber Debiased logits ---
		# anchor: u_k (real user, non-dominant interest k),
		# positive: I_k (global prototype), negative: u_echo_k (echo-chamber user, same k)
		# Computed for all K-1 non-dominant interests simultaneously -> [B*(K-1), 2]
		echo_logits = self._echo_logits(multi_rep, echo_rep, dominant_indices)

		return proto_logits, echo_logits
	
# intiliases various components of the model
	# news_encoder - for encoding news articles
	# multi_rep_encoder - for creating multi_reprs
	# multi_rep_user_encoder - for encoding user repr, and other parameters

# forward
	# i/p - candidate + historical news titles and abstracts + titles and abstracts of neighboring news articles
	# candidate and historical news rep -> passed through news encoder
	# att mech applied to historical news repr to get a refined repr (target_his_rep)
	# historical news rep passed through multi_rep_encoder and multi_rep_user_encoder - to obtain multi-repr of historical news (target_user_rep)
	#model then aggregates info from cand news and target user repr based on agg_mode
	# results stored in predict_logits
	# incorporates GNN to consider information from neighboring news articles (nei1 nei2)
	# constrastive loss using InfoNCE (Noise Contrastive Estimation) between predicted logits and user repr

class Multi_Rep_Predictor(nn.Module):
	def __init__(self, num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode, gnn_mode, agg_mode, temperature=0.07):
		super().__init__()
		self.news_encoder = News_Encoder(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, dropout_rate)
		self.attention = MultiHeadAttention(num_head, hid_dim, hid_dim, dropout_rate)
		self.multi_rep_encoder = Multi_Rep_Encoder(hid_dim, num_prototype, multi_rep_mode, dropout_rate)
		self.user_encoder = Multi_Rep_User_Encoder(num_head, hid_dim, dropout_rate, num_prototype)
		self.prototype = self.multi_rep_encoder.prototype
		self.contrastive_mode = contrastive_mode
		self.gnn_mode = gnn_mode
		self.agg_mode = agg_mode
		self.num_prototype = num_prototype
		self.hid_dim = hid_dim
		# temperature controls sharpness of contrastive distribution (default 0.07 per SimCLR)
		self.infoNCE = InfoNCE(hid_dim, infonce_mode, self.prototype, temperature=temperature)
				
		self.W = nn.Parameter(torch.Tensor(2 * hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.W.data, gain = 1.414)

		self.w1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w1.data, gain = 1.414)
		self.w2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.w2.data, gain = 1.414)
		self.w3 = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
		nn.init.xavier_uniform_(self.w3.data, gain = 1.414)
		self.w4 = nn.Parameter(torch.Tensor(num_prototype, hid_dim, 1))
		nn.init.xavier_uniform_(self.w4.data, gain = 1.414)
		self.w5 = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(200, 1))
		nn.init.xavier_uniform_(self.w5.data, gain = 1.414)
		nn.init.xavier_uniform_(self.proj.data, gain = 1.414)
		#self.wgcn1 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		#self.wgcn2 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		#self.wgcn3 = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		#nn.init.xavier_uniform_(self.wgcn1.data, gain = 1.414)
		#nn.init.xavier_uniform_(self.wgcn2.data, gain = 1.414)
		#nn.init.xavier_uniform_(self.wgcn3.data, gain = 1.414)

	def forward(self, candidate_title, candidate_abstract, his_title, his_abstract,
	            echo_his_title=None, echo_his_abstract=None):
		# echo_his_title / echo_his_abstract: optional echo-chamber augmented history
		# shape identical to his_title / his_abstract: [batch, 50, word_len]
		# only provided when infonce_mode == 'echo_chamber_debiased'
		batch_size = candidate_title.size(0)

		candidate_rep = self.news_encoder(candidate_title, candidate_abstract)	  # [30, 5, 400]
		target_his_rep = self.news_encoder(his_title, his_abstract)    # [30, 50, 400]
		target_his_rep = self.attention(target_his_rep.unsqueeze(dim = 0), target_his_rep.unsqueeze(dim = 0), target_his_rep.unsqueeze(dim = 0)).squeeze(dim = 0)

		# save raw encoded history (before multi_rep_encoder) for dominant prototype computation
		# shape: [B, 50, 400] — mean over articles gives [B, 400] user centroid
		raw_his_rep = target_his_rep

		target_his_rep = self.multi_rep_encoder(target_his_rep)  
		#target_user_rep = torch.mean(target_his_rep, dim = 1)
		target_user_rep = self.user_encoder(target_his_rep)   
		
		if self.agg_mode == 'soft':
			# find similarity between candidate and interest prototypes - loss function improves them in every iteration
			local_att = torch.matmul(candidate_rep, self.prototype.transpose(-2, -1))	 # [30, 5, 10]
			# probability of it being associated with each prototype
			local_att = F.softmax(local_att, dim = 2)
			# user rep wrt candidate articles = delk x uk - prob of being associated with Ik x user rep wrt Ik - so we obtain a repr of the candidate article in terms of the user interest-prototype repr
			local_user_rep = torch.matmul(local_att, target_user_rep)	 # [30, 5, 400]
			# click prob - uT.candidate 
			predict_logits = torch.matmul(candidate_rep.unsqueeze(dim = 2), local_user_rep.unsqueeze(dim = 3))
			# 5 scores for every user in the batch of 30 users (for 5 candidate articles)
			predict_logits = predict_logits.reshape(predict_logits.size(0), predict_logits.size(1))    # [30, 5]
		else:
			candidate_rep_ = self.multi_rep_encoder(candidate_rep).reshape(batch_size, candidate_rep.size(1), self.num_prototype * self.hid_dim)	# [30, 5, 4000]
			target_user_rep_ = target_user_rep.reshape(batch_size, self.num_prototype * self.hid_dim).unsqueeze(dim = 2)	# [30, 4000, 1]
			predict_logits = torch.matmul(candidate_rep_, target_user_rep_).squeeze(dim = 2)

		if self.gnn_mode == 'mgat1':
			pass


		elif self.gnn_mode == 'mgat2':
			pass

		elif self.gnn_mode == 'sgat':
			pass


		elif self.gnn_mode == 'sgcn':
			pass

		else:
			pass

		if self.contrastive_mode == 'USER' and self.training:
			# --- Always compute BOTH contrastive losses during training ---
			# Step 1: encode the echo-chamber augmented history through the same pipeline as real history
			echo_his_rep = self.news_encoder(echo_his_title, echo_his_abstract)   # [B, 50, D]
			echo_his_rep = self.attention(
				echo_his_rep.unsqueeze(0), echo_his_rep.unsqueeze(0), echo_his_rep.unsqueeze(0)
			).squeeze(0)                                                           # [B, 50, D]
			echo_his_rep  = self.multi_rep_encoder(echo_his_rep)                   # [B, 50, K, D]
			echo_user_rep = self.user_encoder(echo_his_rep)                        # [B, K, D]

			# Step 2: identify dominant prototype per user from real history
			# mean-pool raw encoded history [B, 50, D] -> [B, D], find closest prototype -> [B]
			# dominant interest is excluded as anchor in the echo_chamber_debiased loss
			mean_his            = raw_his_rep.mean(dim=1)                          # [B, D]
			similarity_to_proto = torch.matmul(mean_his, self.prototype.t())        # [B, K]
			dominant_indices    = similarity_to_proto.argmax(dim=1)                # [B]

			# Step 3: InfoNCE always computes both logits and returns (proto_logits, echo_logits)
			proto_logits, echo_logits = self.infoNCE(
				target_user_rep,
				echo_rep=echo_user_rep,
				dominant_indices=dominant_indices
			)
			# return 3-tuple: main.py computes loss = pred + alpha*proto + beta*echo
			return predict_logits, proto_logits, echo_logits
		else:
			# eval mode: contrastive logits not needed, return None placeholders
			return predict_logits, None, None

