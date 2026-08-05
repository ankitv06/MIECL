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
	"""
	Contrastive learning module supporting 4 modes:
	  prototype_self        - original MIECL user-prototype CL only
	  echo_chamber_debiased - prototype_self + echo-chamber hard negative CL
	  popularity_debiased   - prototype_self + popularity debiasing news-level CL
	  all_combined          - prototype_self + echo_chamber + popularity

	prototype_self always runs first in every mode.
	Returns a dict: {'proto': Tensor[B,2], 'echo': Tensor|None, 'pop': Tensor|None}
	"""
	def __init__(self, hid_dim, infonce_mode, prototype,
				 temp_proto=0.1, temp_echo=0.07, temp_pop=0.2):
		super().__init__()
		self.mode       = infonce_mode
		self.prototype  = prototype
		self.temp_proto = temp_proto  # temperature for prototype_self path
		self.temp_echo  = temp_echo   # temperature for echo_chamber_debiased path
		self.temp_pop   = temp_pop    # temperature for popularity_debiased path

		self.W = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		nn.init.xavier_uniform_(self.W, gain=1.414)

	# --- prototype_self helper ---
	# anchor: u_k (random user interest), positive: I_k (global prototype),
	# negative: u_k' (different random interest of same user)
	def _proto_logits(self, multi_rep):
		dev = multi_rep.device
		K   = multi_rep.size(1)
		positive_index = torch.randint(low=0, high=K, size=(1,), device=dev)
		negative_index = torch.randint(low=0, high=K, size=(1,), device=dev)
		while positive_index.item() == negative_index.item():
			negative_index = torch.randint(low=0, high=K, size=(1,), device=dev)

		anchor   = torch.index_select(multi_rep, dim=1, index=positive_index).squeeze(dim=1)  # [B, D]
		positive = torch.index_select(self.prototype, dim=0, index=positive_index)             # [1, D]
		negative = torch.index_select(multi_rep, dim=1, index=negative_index).squeeze(dim=1)  # [B, D]

		# L2 normalize -> cosine similarity, scale by temperature
		anchor_n   = F.normalize(anchor,   p=2, dim=-1)
		positive_n = F.normalize(positive, p=2, dim=-1)
		negative_n = F.normalize(negative, p=2, dim=-1)

		pos_logit = torch.matmul(anchor_n, positive_n.t()) / self.temp_proto        # [B, 1]
		neg_logit = (anchor_n * negative_n).sum(dim=-1, keepdim=True) / self.temp_proto  # [B, 1]
		return torch.cat([pos_logit, neg_logit], dim=-1)  # [B, 2]

	# --- echo_chamber_debiased helper ---
	# For every non-dominant interest k, compute:
	#   anchor: u_k (real user interest k),
	#   positive: I_k (global prototype k),
	#   negative: u_echo_k (echo-chamber user, same k)
	# Returns [B*(K-1), 2] — one row per non-dominant interest per sample.
	def _echo_logits(self, multi_rep, echo_rep, dominant_indices):
		B, K, D = multi_rep.size()
		dev = multi_rep.device

		# all_logits: [B*K, 2]
		all_anchor   = multi_rep.reshape(B * K, D)   # [B*K, D]
		all_echo     = echo_rep.reshape(B * K, D)    # [B*K, D]

		# prototype index per interest slot: [0,1,...,K-1] repeated B times
		proto_idx = torch.arange(K, device=dev).unsqueeze(0).expand(B, K).reshape(B * K)  # [B*K]
		all_positive = self.prototype[proto_idx]  # [B*K, D]

		# L2 normalize
		a_n = F.normalize(all_anchor,   p=2, dim=-1)
		p_n = F.normalize(all_positive, p=2, dim=-1)
		n_n = F.normalize(all_echo,     p=2, dim=-1)

		pos_logit = (a_n * p_n).sum(dim=-1, keepdim=True) / self.temp_echo  # [B*K, 1]
		neg_logit = (a_n * n_n).sum(dim=-1, keepdim=True) / self.temp_echo  # [B*K, 1]
		all_logits = torch.cat([pos_logit, neg_logit], dim=-1)               # [B*K, 2]

		# Build mask: True for non-dominant interest slots
		# dominant_indices: [B] — index of dominant interest per sample
		dominant_flat = dominant_indices.unsqueeze(1)  # [B, 1]
		slot_ids = torch.arange(K, device=dev).unsqueeze(0).expand(B, K)  # [B, K]
		non_dominant_mask = (slot_ids != dominant_flat).reshape(B * K)     # [B*K] bool

		return all_logits[non_dominant_mask]  # [B*(K-1), 2]

	# --- popularity_debiased helper ---
	# anchor: popular article rep, positive: unpopular same-topic rep,
	# negative: different-topic article rep
	# news_pair = (popular_rep [B,D], unpopular_rep [B,D], diff_rep [B,D])
	# Returns [B, 2]; sentinel rows (pop_id==0) are zeroed by caller.
	def _pop_logits(self, news_pair):
		popular_rep, unpopular_rep, diff_rep = news_pair
		anchor   = F.normalize(popular_rep,   p=2, dim=-1)
		positive = F.normalize(unpopular_rep, p=2, dim=-1)
		negative = F.normalize(diff_rep,      p=2, dim=-1)
		pos_logit = (anchor * positive).sum(dim=-1, keepdim=True) / self.temp_pop  # [B, 1]
		neg_logit = (anchor * negative).sum(dim=-1, keepdim=True) / self.temp_pop  # [B, 1]
		return torch.cat([pos_logit, neg_logit], dim=-1)  # [B, 2]

	def forward(self, multi_rep, echo_rep=None, dominant_indices=None, news_pair=None):
		"""
		Args:
		  multi_rep         : [B, K, D]  real user multi-interest representations
		  echo_rep          : [B, K, D]  echo-chamber user representations (required for echo/all modes)
		  dominant_indices  : [B]        index of dominant interest per sample (required for echo/all modes)
		  news_pair         : tuple(popular_rep, unpopular_rep, diff_rep) each [B, D]
		                       (required for popularity/all modes; sentinel rows pre-zeroed by caller)
		Returns:
		  dict with keys 'proto', 'echo', 'pop' — values are logit tensors or None
		"""
		# prototype_self always runs in every mode
		result = {
			'proto': self._proto_logits(multi_rep),  # [B, 2]
			'echo' : None,
			'pop'  : None,
		}

		if self.mode in ('echo_chamber_debiased', 'all_combined'):
			assert echo_rep is not None and dominant_indices is not None, \
				"echo_rep and dominant_indices required for mode '{}'".format(self.mode)
			result['echo'] = self._echo_logits(multi_rep, echo_rep, dominant_indices)  # [B*(K-1), 2]

		if self.mode in ('popularity_debiased', 'all_combined'):
			assert news_pair is not None, \
				"news_pair required for mode '{}'".format(self.mode)
			result['pop'] = self._pop_logits(news_pair)  # [B, 2]

		return result
	
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
	def __init__(self, num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix,
				 num_prototype, dropout_rate, multi_rep_mode, infonce_mode, contrastive_mode,
				 gnn_mode, agg_mode, temp_proto=0.1, temp_echo=0.07, temp_pop=0.2):
		super().__init__()
		self.news_encoder      = News_Encoder(num_head, hid_dim, word_dim, word_matrix, entity_dim, entity_matrix, dropout_rate)
		self.attention         = MultiHeadAttention(num_head, hid_dim, hid_dim, dropout_rate)
		self.multi_rep_encoder = Multi_Rep_Encoder(hid_dim, num_prototype, multi_rep_mode, dropout_rate)
		self.user_encoder      = Multi_Rep_User_Encoder(num_head, hid_dim, dropout_rate, num_prototype)
		self.prototype         = self.multi_rep_encoder.prototype
		self.contrastive_mode  = contrastive_mode
		self.infonce_mode      = infonce_mode
		self.gnn_mode          = gnn_mode
		self.agg_mode          = agg_mode
		self.num_prototype     = num_prototype
		self.hid_dim           = hid_dim
		self.infoNCE = InfoNCE(hid_dim, infonce_mode, self.prototype,
							   temp_proto=temp_proto, temp_echo=temp_echo, temp_pop=temp_pop)

		self.W    = nn.Parameter(torch.Tensor(2 * hid_dim, hid_dim))
		self.w1   = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w2   = nn.Parameter(torch.Tensor(hid_dim, hid_dim))
		self.w3   = nn.Parameter(torch.Tensor(num_prototype, hid_dim))
		self.w4   = nn.Parameter(torch.Tensor(num_prototype, hid_dim, 1))
		self.w5   = nn.Parameter(torch.Tensor(hid_dim, 200))
		self.proj = nn.Parameter(torch.Tensor(200, 1))
		for p in [self.W, self.w1, self.w2, self.w3, self.w4, self.w5, self.proj]:
			nn.init.xavier_uniform_(p.data, gain=1.414)

	# shared helper: encode a news history through news_encoder -> attention -> multi_rep -> user_encoder
	def _encode_history(self, his_title, his_abstract):
		his_rep = self.news_encoder(his_title, his_abstract)  # [B, 50, D]
		his_rep = self.attention(
			his_rep.unsqueeze(0), his_rep.unsqueeze(0), his_rep.unsqueeze(0)
		).squeeze(0)                                          # [B, 50, D]
		his_rep = self.multi_rep_encoder(his_rep)             # [B, 50, K, D]
		return self.user_encoder(his_rep)                     # [B, K, D]

	def forward(self, candidate_title, candidate_abstract,
				his_title, his_abstract,
				# echo-chamber debiased inputs (optional)
				echo_his_title=None, echo_his_abstract=None,
				# popularity debiased inputs (optional)
				pop_title=None,   pop_abstract=None,
				unpop_title=None, unpop_abstract=None,
				diff_title=None,  diff_abstract=None):
		batch_size = candidate_title.size(0)

		# --- Step 1: Encode candidate articles ---
		candidate_rep = self.news_encoder(candidate_title, candidate_abstract)  # [B, 5, D]

		# --- Step 2: Encode real user history ---
		target_user_rep = self._encode_history(his_title, his_abstract)         # [B, K, D]

		# --- Step 3: Predict click scores ---
		if self.agg_mode == 'soft':
			local_att      = F.softmax(torch.matmul(candidate_rep, self.prototype.t()), dim=2)  # [B, 5, K]
			local_user_rep = torch.matmul(local_att, target_user_rep)                           # [B, 5, D]
			predict_logits = torch.matmul(
				candidate_rep.unsqueeze(2), local_user_rep.unsqueeze(3)
			).reshape(batch_size, candidate_rep.size(1))                                        # [B, 5]
		else:
			cand_  = self.multi_rep_encoder(candidate_rep).reshape(
				batch_size, candidate_rep.size(1), self.num_prototype * self.hid_dim)            # [B, 5, K*D]
			user_  = target_user_rep.reshape(
				batch_size, self.num_prototype * self.hid_dim).unsqueeze(2)                      # [B, K*D, 1]
			predict_logits = torch.matmul(cand_, user_).squeeze(2)                              # [B, 5]

		# GNN placeholder (no-op in current codebase)
		if self.gnn_mode not in ('mgat1', 'mgat2', 'sgat', 'sgcn', 'nogat'):
			pass

		# --- Step 4: Contrastive learning paths ---
		if self.contrastive_mode != 'USER' or not self.training:
			# eval mode or no contrastive: return empty dict
			return predict_logits, {'proto': None, 'echo': None, 'pop': None}

		# --- 4a: Echo-chamber: encode echo history ---
		echo_rep        = None
		dominant_indices = None
		if echo_his_title is not None and self.infonce_mode in ('echo_chamber_debiased', 'all_combined'):
			echo_user_rep = self._encode_history(echo_his_title, echo_his_abstract)   # [B, K, D]
			echo_rep      = echo_user_rep                                               # [B, K, D]
			# dominant interest = prototype with highest user attention score
			with torch.no_grad():
				mean_user_rep = target_user_rep.mean(dim=1)  # [B, D]
				scores = torch.matmul(
					F.normalize(mean_user_rep, p=2, dim=-1),
					F.normalize(self.prototype,  p=2, dim=-1).t()
				)  # [B, K]
				dominant_indices = scores.argmax(dim=1)  # [B]

		# --- 4b: Popularity: encode triplet, compute reps, apply sentinel mask ---
		news_pair = None
		if pop_title is not None and self.infonce_mode in ('popularity_debiased', 'all_combined'):
			popular_rep  = self.news_encoder(pop_title,   pop_abstract).squeeze(1)    # [B, D]
			unpopular_rep = self.news_encoder(unpop_title, unpop_abstract).squeeze(1)  # [B, D]
			diff_rep     = self.news_encoder(diff_title,  diff_abstract).squeeze(1)   # [B, D]

			# Sentinel mask: rows where pop_id == 0 have all-zero popular_rep -> zero them out
			# so they contribute no gradient (handled in main.py before loss calc)
			valid_mask   = (popular_rep.abs().sum(dim=-1) > 0).float().unsqueeze(-1)   # [B, 1]
			popular_rep  = popular_rep  * valid_mask
			unpopular_rep = unpopular_rep * valid_mask
			diff_rep     = diff_rep     * valid_mask
			news_pair    = (popular_rep, unpopular_rep, diff_rep)

		# --- 4c: Call InfoNCE and return dict ---
		logits_dict = self.infoNCE(
			target_user_rep,
			echo_rep=echo_rep,
			dominant_indices=dominant_indices,
			news_pair=news_pair,
		)
		return predict_logits, logits_dict

