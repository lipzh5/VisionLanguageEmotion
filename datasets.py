# -*- coding:utf-8 -*-
import os
import json
import glob
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import torch
import torchvision.transforms as transforms
import pandas as pd
import random
import librosa
import vocab
# from collections import defaultdict

from conf.config import *
import os.path as osp
from tqdm import tqdm
import cv2
from PIL import Image
# from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor
from transformers import AutoTokenizer, AutoProcessor #, AutoImageProcessor, Data2VecAudioModel
# from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model # Data2VecAudioModel


torch.set_default_dtype(torch.float32)
transform, resize = None, None
tokenizer = None
# audio_processor, tokenizer = None, None



# def get_audio_processor(pretrained_path, model_name='data2vec'):
# 	global audio_processor
# 	if audio_processor is None:
# 		# audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
# 		audio_processor = AutoProcessor.from_pretrained(pretrained_path) if model_name == 'data2vec' else Wav2Vec2FeatureExtractor.from_pretrained(pretrained_path)
# 	return audio_processor
   

def get_tokenizer(pretrained_path):
	global tokenizer
	if tokenizer is None:
		tokenizer = AutoTokenizer.from_pretrained(pretrained_path, local_files_only=False)
		_special_tokens_ids = tokenizer('<mask>')['input_ids']
		CLS = _special_tokens_ids[0]
		MASK = _special_tokens_ids[1]
		SEP = _special_tokens_ids[2]
		CONFIG['CLS'] = CLS  
		CONFIG['SEP'] = SEP
		CONFIG['mask_value'] = MASK
	return tokenizer



class Transform:
	def __init__(self, cfg, split_type):  # cfg: config.data.transform
		transform_list = [transforms.ColorJitter(**cfg.color_jitter)] if split_type == 'train' else []
		self.transform = transforms.Compose(
			transform_list +
			[transforms.ToTensor(), transforms.Normalize(mean=cfg.mean, std=cfg.std)])  

	def __call__(self, data):
		return self.transform(data)


class Resize:
	def __init__(self, target_size):
		self.target_size = target_size  # cfg.resize.target_size

	def __call__(self, img: np.ndarray):
		interp = cv2.INTER_AREA if img.shape[0] > self.target_size else cv2.INTER_CUBIC
		return cv2.resize(img, dsize=(self.target_size, self.target_size), interpolation=interp)


def get_transform(cfg, split_type):
	global transform
	if transform is None:
		transform = Transform(cfg, split_type)
	return transform

def get_resize(target_size=160): # for InceptionResnetV1
	global resize
	if resize is None:
		resize = Resize(target_size=160)
	return resize
		


def preprocess_input_vision_encoder(image_path_list, cfg, split_type, no_padding=False, no_resize=False, neutral_norm=False, neutral_face_path=None, filter_masked_openface=False):
	transform = get_transform(cfg, split_type)
	resize = get_resize()

	img_mask = torch.zeros((VISION_MAX_UTT_LEN,))
	# len_images = len(image_path_list)
	# img_mask[:len_images] = 1
	im_list = []

	# filter_masked_openface = cfg.train.vfeat_filter_masked_openface   # filterout all zero images (no landmark detected)

	if neutral_norm:  # normalize using neural frame
		# img_mask = img_mask[1:]   # drop the first ref frame
		if neutral_face_path is not None:
			ref_frame = cv2.imread(neutral_face_path)
			# ref_frame = resize(ref_frame)
		else:
			ref_frame = cv2.imread(image_path_list[0])  # [0, 255]
		# im_list.append(transform(Image.fromarray(resize(ref_frame), mode='RGB')))
		# img_mask[0] = 1 if len_images <= 1 else 0
		for img_path in image_path_list:
			im = cv2.imread(img_path)
			if im is None:
				continue
			if ref_frame.shape != im.shape:
				ref_frame = resize(ref_frame)
			im = im - ref_frame
			# cv2.imwrite('debug_delta.png', im)
			im_list.append(transform(Image.fromarray(resize(im), mode='RGB')))
	else:
		for img_path in image_path_list:
			im = cv2.imread(img_path)
			if im is None:
				continue
			if filter_masked_openface and im.sum() < 32 and len(im_list) > 0:  # TODO magic number
				continue
			if no_resize:
				im_list.append(transform(Image.fromarray(im, mode='RGB')))
			else:
				im_list.append(transform(Image.fromarray(resize(im), mode='RGB')))
	img_inputs = torch.stack(im_list, dim=0)
	len_images = len(im_list)
	img_mask[:len_images] = 1
	
	if no_padding:
		return img_inputs, img_mask
	img_inputs = torch.concat((img_inputs, torch.zeros((VISION_MAX_UTT_LEN-len_images, 3, RESNET_IMG_SIZE, RESNET_IMG_SIZE))))

	# print(img_inputs.requires_grad, img_inputs.shape) # [174, 3, 160, 160]
	return img_inputs, img_mask

	
def pad_to_len(sequence_data, max_len, pad_value):
	sequence_data = sequence_data[-max_len:]
	effective_len = len(sequence_data)
	mask = torch.zeros((max_len,))
	mask[:effective_len] = 1

	len_to_pad = max_len - effective_len
	pads = [pad_value]*len_to_pad
	if isinstance(sequence_data, list):
		sequence_data.extend(pads)
	elif isinstance(sequence_data, torch.Tensor):
		sequence_data = torch.concat((sequence_data, torch.tensor(pads)))
   
	return sequence_data, mask
	# if return_padding_len:
	#     return sequence_data, len_to_pad
	# return sequence_data


def get_meld_vocabs(anno_csv_dir, vocab_path):
	file_paths = [osp.join(anno_csv_dir, f'{split_type}_sent_emo.csv') for split_type in ('train', 'val', 'test')]
	emotion_vocab = vocab.Vocab()
	# keep 'neutral' in index-0
	emotion_vocab.word2index('neutral', train=True)
	# global speaker_vocab, emotion_vocab
	for file_path in file_paths:
		data = pd.read_csv(file_path)
		for row in tqdm(data.iterrows(),
						desc='get vocab from {}'.format(file_path)): # disable=CONFIG['local_rank'] not in [-1, 0]
			meta = row[1]
			emotion = meta['Emotion'].lower()
			emotion_vocab.word2index(emotion, train=True)
	# if CONFIG['local_rank'] in [-1, 0]:
	torch.save(emotion_vocab.to_dict(), vocab_path)
	logging.info('total {} emotions'.format(len(emotion_vocab)))

   

def load_meld_turn(anno_csv_dir, split_type, vocab_path):
	
	emotion_vocab = vocab.Vocab.from_dict(torch.load(vocab_path))
	file_path = osp.join(anno_csv_dir, f'{split_type}_sent_emo.csv')
	data = pd.read_csv(file_path)
	pre_dial_id = -1
	dialogues = []
	dialogue = []
	speaker_vocab = vocab.Vocab()
	for row in tqdm(data.iterrows(),
					desc='processing file {}'.format(file_path)): # disable=CONFIG['local_rank'] not in [-1, 0]
		meta = row[1]
		text = meta['Utterance'].replace('’', '\'').replace("\"", '')
		speaker = meta['Speaker']
		emotion = meta['Emotion'].lower()
		emotion_idx = emotion_vocab.word2index(emotion)
		turn_data = {}
		turn_data['speaker'] = speaker
		speaker_vocab.word2index(speaker, train=True)
		turn_data['text'] = text
		turn_data['label'] = emotion_idx

		dialogue_id = meta['Dialogue_ID']
		if pre_dial_id == -1:
			pre_dial_id = dialogue_id
		if dialogue_id != pre_dial_id:
			dialogues.append(dialogue)
			dialogue = []
		pre_dial_id = dialogue_id
		dialogue.append(turn_data)
	dialogues.append(dialogue)  # 13707 + 260

	# speaker_vocab = speaker_vocab.prune_by_count(30)
	# for speaker_name in speaker_vocab.counts.keys():
	#     tokenizer.add_tokens(speaker_name)
	return dialogues



def get_openface_aligned_img_paths(anno_csv_path, split_type, img_paths_save_to='openface_img_paths_train.json'):
	if osp.exists(img_paths_save_to):
		with open(img_paths_save_to, 'r') as fp:
			data = json.load(fp)
		return data

	img_paths = {}
	face_dir = osp.join(anno_csv_path, f'raw/MELD.Raw/openfacefeat_{split_type}_')
	utt_names = os.listdir(face_dir)   # 9989
	for utt_name in utt_names:
		# img_dir = osp.join(face_dir, utt_name, f'{utt_name}_aligned')
		img_dir = osp.join(face_dir, utt_name)
		try:
			imgs = sorted(os.listdir(img_dir))
			img_paths[utt_name] = [osp.join(img_dir, img_name) for img_name in imgs]
		except Exception as e:
			print(str(e))
	
	with open(img_paths_save_to, 'w') as fp:
		json.dump(img_paths, fp)

	return img_paths


def get_utt_to_speaker(anno_csv_path, split_type): 
	'''for leading roles only'''
	file_path = osp.join(anno_csv_path, f'{split_type}_sent_emo.csv')
	data = pd.read_csv(file_path)
	utt_to_speaker = {}
	for row in tqdm(data.iterrows(), desc=f'audio processing {file_path}'):
		meta = row[1]
		utt_name = f"dia{meta['Dialogue_ID']}_utt{meta['Utterance_ID']}"
		speaker_name = meta['Speaker'].lower()
		if speaker_name in LEARDING_ROLES:
			utt_to_speaker[utt_name] = speaker_name
	return utt_to_speaker
	 

# input faceseq directly and adapt face embedding during training
class MultimodalDataset(Dataset):   
	"""loading multimodal data"""
	# TODO can shuffle dialogues
	def __init__(self, cfg, split_type):
		super(MultimodalDataset, self).__init__()

		self.split_type = split_type
		self.cwd = os.getcwd()
		
		# int64
		# text_inputs: (input_ids, labels, mask)
		# self.all_utt_idx_with_extra = []  # utt_idx in the sequential data list
		anno_csv_path = cfg.dataset.meld.anno_csv_path
		self.init_text_with_context_modeling(
			anno_csv_path, split_type, cfg.dataset.meld.emotion_vocab_path, cfg.model.text_encoder.pretrained_path, 
			cfg.data.context_max_len, cfg.data.context_pad_value)  # init labels as well
		# print(f'shape text input ids: {self.text_input_ids.shape}, labels shape: {self.text_labels_for_extra.shape}')  # (1109, 256)
		
		# record utt-speaker mapping
		self.utt_to_speaker = get_utt_to_speaker(anno_csv_path, split_type)
		'''visual modality inputs
		   TODO extract from raw video
		'''
		self.setup_vision_feat_fetcher(cfg)

		self.vfeat_neutral_norm = cfg.train.vfeat_neutral_norm
		self.neutral_face_path = cfg.dataset.meld.neutral_face_path

		# if cfg.train.vfeat_penny == 3 and not cfg.train.vfeat_from_pkl:  # using imgs obtained from openface
		save_to = osp.join(anno_csv_path, f'preprocessed_data/vision/openface_img_paths_{split_type}.json')
		self.openface_img_paths = get_openface_aligned_img_paths(anno_csv_path, split_type, save_to)
		print(f'openface img path: {save_to} \n *****')
		

	def setup_vision_feat_fetcher(self, cfg):
		dataset_path = cfg.dataset.data_load_path
		self.data_transform_cfg = cfg.data.transform
		utt_profile_path = os.path.join(dataset_path, 'text', f'{self.split_type}_utt_profile.json')
		with open(utt_profile_path, 'r') as rd:
			self.utt_profile = json.load(rd)   # 'idx': ['utt_name', 'dia-name',...]
		self.vision_feat_fetcher = self._fetch_vision_feat_from_processor_openface
		

	def init_text_with_context_modeling(self, anno_csv_dir, split_type, vocab_path, pretrained_path, max_len, pad_value):
		 # construct additional training data for other modalities (TODO : construct from raw input)
		# dialogues_path = osp.join(anno_csv_dir, f'{split_type}_dialogues.pkl')
		# dialogues = torch.load(dialogues_path) if osp.exists(dialogues_path) else load_meld_turn(anno_csv_dir, split_type, vocab_path)
		dialogues = load_meld_turn(anno_csv_dir, split_type, vocab_path)

		ret_utterances = []
		ret_labels = []
		tokenizer = get_tokenizer(pretrained_path)  # len: 50265
		for dialogue in dialogues:
			utterance_ids = []
			query = 'For utterance:'
			query_ids = tokenizer(query)['input_ids'][1:-1]
			for idx, turn_data in enumerate(dialogue):
				text_with_speaker = turn_data['speaker'] + ':' + turn_data['text']
				token_ids = tokenizer(text_with_speaker)['input_ids'][1:]
				utterance_ids.append(token_ids)
				if turn_data['label'] < 0:
					continue
				full_context = [CONFIG['CLS']]
				lidx = 0
				for lidx in range(idx):
					total_len = sum([len(item) for item in utterance_ids[lidx:]]) + 8
					if total_len + len(utterance_ids[idx]) <= max_len: # CONFIG['max_len']:
						break
				lidx = max(lidx, idx-8)
				for item in utterance_ids[lidx:]:
					full_context.extend(item)

				query_idx = idx
				prompt = dialogue[query_idx]['speaker'] + ' feels <mask>'
				full_query = query_ids + utterance_ids[query_idx] + tokenizer(prompt)['input_ids'][1:]
				input_ids = full_context + full_query
				input_ids, _ = pad_to_len(input_ids, max_len, pad_value) # CONFIG['max_len'], CONFIG['pad_value']
				ret_utterances.append(input_ids)
				ret_labels.append(dialogue[query_idx]['label'])

	
		self.text_input_ids = torch.tensor(ret_utterances, dtype=torch.long)
		self.labels = torch.tensor(ret_labels, dtype=torch.long)

	
	
	def __len__(self):
		return len(self.labels)
		# return len(self.all_utt_idx_with_extra)

	def _fetch_audio_feat_from_pkl(self, index):  # .shape为(utt_max_lens, Feature_extra_dim) 
		return self.audio_feature[index], self.audio_utterance_mask[index]  

	def _fetch_audio_feat_from_processor(self, index):
		return self.audio_ivalues['audio_ivalues'][index], self.audio_ivalues['audio_mask'][index]

	def _fetch_vision_feat_from_pkl(self, index):
		return self.vision_feature[index], self.vision_utterance_mask[index]
	

	def _fetch_vision_feat_from_processor_openface(self, index):
		curr_utt_name, curr_dia_name, dia_idx, curr_dia_len,  curr_idx_in_dia = self.utt_profile[str(index)]
		curr_utt_frm_list = self.openface_img_paths[curr_utt_name][:VISION_MAX_UTT_LEN] # sequence of image path
		# if self.vfeat_downsample_to > 0:
		# 	curr_utt_frm_list = downsample_images(curr_utt_frm_list)

		speaker_name=self.utt_to_speaker.get(curr_utt_name)
		neutral_face_path = osp.join(self.neutral_face_path, f'{speaker_name}_neutral.bmp') if speaker_name else None
		
		return preprocess_input_vision_encoder(
			curr_utt_frm_list, self.data_transform_cfg, split_type=self.split_type, 
			no_resize=False, neutral_norm=self.vfeat_neutral_norm, neutral_face_path=neutral_face_path)
	


	def __getitem__(self, index):    
		curr_text_input_ids = self.text_input_ids[index]
		curr_label_ids = self.labels[index]

		'''vision'''
		vision_inputs, vision_mask = self.vision_feat_fetcher(index)
		return curr_text_input_ids, vision_inputs, vision_mask, curr_label_ids
