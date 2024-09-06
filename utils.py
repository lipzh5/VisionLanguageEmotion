# -*- coding:utf-8 -*-
import os
import os.path as osp
import random
import torch
import numpy as np
from sklearn.metrics import f1_score
from datasets import MultimodalDataset

from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import shutil
import torch.nn.functional as F
from models.vision_encoders import SupConResNet
import logging
log = logging.getLogger(__name__)


class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1, accu_steps=1):
		self.val = val
		self.sum += val * n * accu_steps
		self.count += n
		self.avg = self.sum / self.count



def eval_meld(results, truths, test=False, return_all=False):
	test_preds = results.cpu().detach().numpy()   #（num_utterance, num_label)
	test_truth = truths.cpu().detach().numpy()  #（num_utterance）
	predicted_label = []
	true_label = []
	for i in range(test_preds.shape[0]):
		predicted_label.append(np.argmax(test_preds[i,:],axis=0) ) #
		true_label.append(test_truth[i])
	wg_av_f1 = f1_score(true_label, predicted_label, average='weighted')
	# if test:
	f1_each_label = f1_score(true_label, predicted_label, average=None)
	test_str = 'TEST' if test else 'EVAL'
	print(f'**{test_str}** | f1 on each class (Neutral, Surprise, Fear, Sadness, Joy, Disgust, Anger): \n', f1_each_label)
	if return_all:
		return wg_av_f1, f1_each_label
	return wg_av_f1 



def get_multimodal_data(cfg, split_type='train'):
	return MultimodalDataset(cfg, split_type)


def save_model(model, optimizer, cfg, model_name='vle_model.pth'):
	save_path = cfg.train.save_model_path
	if not osp.exists(save_path):
		os.makedirs(save_path)
	if model_name:
		save_path = osp.join(save_path, model_name)
		state = {
			'opt': cfg,
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		torch.save(state, save_path)
		print(f'saved model at: {save_path} \n******')



def set_vision_encoder(cfg):
	pretrained_model_path = cfg.model.vision_encoder.pretrained_path
	if pretrained_model_path:
		resnet = SupConResNet()
		# resnet = SupCEResNet(name='inceptionresnetv1', num_classes=3, use_webface_pretrain=cfg.model.vision_encoder.use_webface_pretrain)
		ckpt = torch.load(pretrained_model_path, map_location='cpu')
		state_dict = ckpt['model']
		device_id = cfg.device_id[0]
		resnet = resnet.cuda(device_id)
		resnet.load_state_dict(state_dict)
		vision_encoder = resnet.encoder
	else:
		use_webface_pretrain = cfg.model.vision_encoder.use_webface_pretrain
		print(f'Inception uses webface pretrain: {use_webface_pretrain} \n ******')
		vision_encoder = InceptionResnetV1(pretrained='casia-webface') if use_webface_pretrain else InceptionResnetV1()

	vis_enc_trainable = cfg.train.resnet_trainable
	vision_encoder.train(vis_enc_trainable)
	vision_encoder.requires_grad_(vis_enc_trainable)
	return vision_encoder


