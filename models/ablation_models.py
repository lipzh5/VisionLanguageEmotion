# -*- coding:utf-8 -*-
# @Desc: None

import torch
import torch.nn as nn
from transformers import AutoModel

import sys
print(f'sys path: {sys.path} \n*****')

from models.modules.transformer import TransformerEncoder, AdditiveAttention
from models.modules.cross_modal_transformer import CrossModalTransformerEncoder
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
log = logging.getLogger(__name__)

from utils import *
from mmcv.cnn import xavier_init

from conf.config import ATTN_MASK_FILL 


class LEModel(nn.Module):
	"""language to emotion"""
	def __init__(self, cfg):
		super().__init__()
		# cross transformer modules
		transformer_conf = cfg.model.transformers
		hidden_size = transformer_conf.hidden_size
		tenc_cfg = cfg.model.text_encoder
		self.text_linear = nn.Linear(tenc_cfg.embed_dim, hidden_size)   

		self.dropout = nn.Dropout(transformer_conf.self_attn_transformer.hidden_dropout_prob)
		self.classifier = nn.Linear(hidden_size, cfg.data.num_labels)

		self._init_weights()
		
		"""text"""
		self.context_encoder = AutoModel.from_pretrained(tenc_cfg.pretrained_path, local_files_only=False)
		self.pad_value = tenc_cfg.pad_value   
		self.mask_value = tenc_cfg.mask_value 

		
	def _init_weights(self):
		# ref to: https://github.com/junjie18/CMT/tree/master
		for m in self.modules():
			if hasattr(m, 'weight') and m.weight.dim() > 1:
				xavier_init(m, distribution='uniform')
		self._is_init = True

	def gen_text_reps(self, sentences, text_mask):
		"""generate vector representation for each turn of conversation"""
		batch_size, max_len = sentences.shape[0], sentences.shape[-1]
		sentences = sentences.reshape(-1, max_len)
		mask = 1 - (sentences == (self.pad_value)).long()
		utterance_encoded = self.context_encoder(
			input_ids=sentences,
			attention_mask=mask,
			output_hidden_states=True,
			return_dict=True
		)['last_hidden_state']
		mask_pos = (sentences == (self.mask_value)).long().max(1)[1]
		mask_outputs = utterance_encoded[torch.arange(mask_pos.shape[0]), mask_pos, :]
		return self.text_linear(mask_outputs) 

	
	def forward(self, text_input_ids, dummy_vision_inputs, dummy_vision_mask):
		text_mask = 1 - (text_input_ids == (self.pad_value)).long()
		text_utt_linear = self.gen_text_reps(text_input_ids, text_mask)
		return self.classifier(self.dropout(text_utt_linear))
	

		

class VEModel(nn.Module):
	"""vision to emotion"""
	def __init__(self, cfg):
		super().__init__()
		# cross transformer modules
		transformer_conf = cfg.model.transformers
		hidden_size = transformer_conf.hidden_size
		tenc_cfg = cfg.model.text_encoder
		self.vision_linear = nn.Linear(cfg.data.vision_feature_dim, hidden_size)
		"""vision"""
		self.vision_utt_transformer = TransformerEncoder(
		transformer_conf.self_attn_transformer,
		transformer_conf.self_attn_transformer.num_transformer_layers.vision,
		cfg.data.vision_utt_max_len, hidden_size)

		self.additive_attn = AdditiveAttention(hidden_size, hidden_size)

		self.dropout = nn.Dropout(transformer_conf.self_attn_transformer.hidden_dropout_prob)
		self.classifier = nn.Linear(hidden_size, cfg.data.num_labels)

		self._init_weights()
		
		self.vision_encoder = set_vision_encoder(cfg)

		
	def _init_weights(self):
		# ref to: https://github.com/junjie18/CMT/tree/master
		for m in self.modules():
			if hasattr(m, 'weight') and m.weight.dim() > 1:
				xavier_init(m, distribution='uniform')
		self._is_init = True

	
	def gen_vision_reps(self, img_inputs, vision_mask): 
		bs, max_utt_img_len, channel, width, height = img_inputs.shape
		img_inputs = img_inputs.reshape(bs*max_utt_img_len, channel, width, height)
		vision_mask = vision_mask.reshape(bs*max_utt_img_len)
		real_img_inputs = img_inputs[vision_mask>0]

		embeddings = self.vision_encoder(real_img_inputs)
		embedding_dim = embeddings.shape[-1]
		output_embeddings = torch.zeros((bs*max_utt_img_len, embedding_dim)).to(img_inputs.device)
		output_embeddings[vision_mask>0] = embeddings
		output_embeddings = output_embeddings.reshape(bs, max_utt_img_len, -1)

		return self.vision_linear(output_embeddings)
	

	
	def forward(self, text_input_ids, vision_inputs, vision_mask):
		
		vision_linear = self.gen_vision_reps(vision_inputs, vision_mask)

		vision_extended_utt_mask = vision_mask.unsqueeze(1).unsqueeze(2)
		vision_extended_utt_mask = (1.0 - vision_extended_utt_mask) * ATTN_MASK_FILL

		vision_utt_trans = self.vision_utt_transformer(vision_linear, vision_extended_utt_mask)
		
		vision_reps, _ = self.additive_attn(vision_utt_trans, vision_mask)
		return self.classifier(self.dropout(vision_reps))
		
