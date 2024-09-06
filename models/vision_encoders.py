# -*- coding:utf-8 -*-

from facenet_pytorch import MTCNN, InceptionResnetV1
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F


def inceptionresnetv1(**kwargs):   # Penny Note: should be consistent with vsupcon
	use_webface_pretrain = kwargs.get('use_webface_pretrain', False)
	return InceptionResnetV1(pretrained='casia-webface') if use_webface_pretrain else InceptionResnetV1()


# import torchvision.models as models


model_dict = {
	'inceptionresnetv1': [inceptionresnetv1, 512],  # model_name: func, hidden_dim 
	# 'resnet18': [models.resnet18, 512],
	# 'resnet34': [models.resnet34, 512],
	# 'resnet50': [models.resnet50, 1000],
	# 'resnet101': [models.resnet101, 2048],
}



# TODO : in and feat dim
class SupConResNet(nn.Module):
	"""backbone + projection head"""
	def __init__(self, name='inceptionresnetv1', head='mlp', feat_dim=512, use_webface_pretrain=True):
		super().__init__()
		model_fn, hidden_dim = model_dict[name]
		self.encoder = model_fn(use_webface_pretrain=use_webface_pretrain)
		if head=='linear':
			self.head = nn.Linear(feat_dim, feat_dim)
		elif head == 'mlp':
			self.head = nn.Sequential(
				nn.Linear(feat_dim, hidden_dim),  
				nn.ReLU(inplace=True),
				nn.Linear(hidden_dim, feat_dim)
			)
		else:
			raise NotImplementedError(f'head not supported: {head}')
	
	def forward(self, x):
		feat = self.encoder(x)
		feat = F.normalize(self.head(feat), dim=1)  # projection head will be discarded during inference
		return feat