# -*- coding:utf-8 -*-

import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import time
from time import strftime
import transformers
import numpy as np

import random
from models.ablation_models import VEModel, LEModel

from torch.utils.tensorboard import SummaryWriter

import logging
log = logging.getLogger(__name__)
from utils import *


def multimodal_train(cfg, train_loader, model, optimizer, scheduler, loss_fn, epoch):
	model.train()
	losses = AverageMeter()
	
	num_batches = len(train_loader)
	device_id = cfg.device_id[0]
	num_epochs = cfg.train.num_epochs
	accu_steps = cfg.train.accumulation_steps
	
	start_time = time.time()
	optimizer.zero_grad()
	for i_batch, batch in enumerate(train_loader):
		batch = [t.cuda(device_id) for t in batch]
		text_input_ids, vision_inputs, vision_mask, label_ids = batch

		logits = model(text_input_ids, vision_inputs, vision_mask)
		loss = loss_fn(logits, label_ids) 
		losses.update(loss.item(), label_ids.shape[0])
		
		loss = loss / accu_steps
		loss.backward()
		
		if ((i_batch + 1) % accu_steps) == 0: # or (i_batch + 1) == len_train_loader: note: may drop grads of the last batch
			torch.nn.utils.clip_grad_norm(model.parameters(), cfg.train.gradient_clip_value)
			optimizer.step()
			scheduler.step()
			optimizer.zero_grad()
		
		if (i_batch + 1) % cfg.train.log_interval == 0:  
			elapsed_time = time.time() - start_time
			log.info(
				f'**TRAIN**|Epoch {epoch}/{num_epochs} | Batch {i_batch + 1}/{num_batches} | Time/Batch(ms) {elapsed_time * 1000.0 / cfg.train.log_interval} | Train Loss {losses.avg}')
			start_time = time.time()
	return losses.avg


def multimodal_evaluate(cfg, data_loader, model, loss_fn, epoch, test=False):
	losses = AverageMeter()
	model.eval()
	res_str = 'test' if test else 'validation'
	results, truths = [], []
	num_batches = len(data_loader)
	num_epochs = cfg.train.num_epochs
	device_id = cfg.device_id[0]

	with torch.no_grad():
		for i_batch, batch in enumerate(data_loader):
			batch = [t.cuda(device_id) for t in batch]
			text_input_ids, vision_inputs, vision_mask, label_ids = batch
			
			logits = model(text_input_ids, vision_inputs, vision_mask)
			loss = loss_fn(logits, label_ids) 

			losses.update(loss.item(), label_ids.shape[0])
	
			results.append(logits)  
			truths.append(label_ids)  
			if (i_batch + 1) % cfg.train.log_interval == 0:
				msg = f'{res_str} res: batch {i_batch + 1}, total loss: {losses.sum}, cur loss: {loss.item()}'
				log.info(msg)

	results = torch.cat(results)
	truths = torch.cat(truths)
	return losses.avg, results, truths

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
	np.random.seed(cfg.seed)
	torch.manual_seed(cfg.seed)
	random.seed(cfg.seed)
	
	wp_rate = cfg.train.warm_up
	vfeat_webface = int(cfg.model.vision_encoder.use_webface_pretrain)
	vfeat_neutral_norm = cfg.train.vfeat_neutral_norm
	use_faceseq160 = int(cfg.train.use_faceseq160)
	vfeat_no_face_ext = int(cfg.train.vfeat_no_face_ext)
	ablation = cfg.train.ablation
	trial_name = f"seed{cfg.seed}_ablation-{ablation}_lr{cfg.train.lr}_decay{cfg.train.weight_decay}_vf-neun{vfeat_neutral_norm}_vf-webface{vfeat_webface}_vf-facesq160{use_faceseq160}_cf-nofaceext{vfeat_no_face_ext}_wp{wp_rate}_bs{cfg.train.batch_size}_accu{cfg.train.accumulation_steps}_ep{cfg.train.num_epochs}_opt-adamw_trial{cfg.trial}"
	writer = SummaryWriter(osp.join('runs',trial_name))
	log.info(f"***********\n TRIAL: {trial_name}\n STARTS!***********")

	device_id = cfg.device_id[0]
	bs_multiplier = 2  
	n_workers = cfg.train.num_workers

	train_data = get_multimodal_data(cfg, 'train')
	print(f'multimodal train data set len: {len(train_data)} \n *****')
	# train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=cfg.train.batch_size, num_workers=n_workers, pin_memory=pin_memory)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=cfg.train.batch_size, num_workers=n_workers)
	val_data = get_multimodal_data(cfg, 'val')
	val_loader = DataLoader(val_data, shuffle=False, batch_size=cfg.train.batch_size * bs_multiplier, num_workers=n_workers)

	test_data = get_multimodal_data(cfg, 'test')
	test_loader = DataLoader(test_data, shuffle=False, batch_size=cfg.train.batch_size * bs_multiplier, num_workers=n_workers)

	# model = VLEModel(cfg).cuda(device_id)
	model = VEModel(cfg) if cfg.train.ablation == 1 else LEModel(cfg)
	model = model.cuda(device_id)

	optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

	num_epochs = cfg.train.num_epochs
	'''cosine schedule with warmup'''                                               
	total_training_steps = num_epochs * len(train_loader)  // cfg.train.accumulation_steps
	scheduler = transformers.get_cosine_schedule_with_warmup(
		optimizer=optimizer,
		num_warmup_steps=int(total_training_steps * cfg.train.warm_up),
		num_training_steps=total_training_steps)
	loss_fn = torch.nn.CrossEntropyLoss().cuda(device_id)
	
	best_val_wf = 0.0
	for epoch in range(1, num_epochs + 1):
	
		ts = time.time()
		train_loss = multimodal_train(cfg, train_loader, model, optimizer, scheduler, loss_fn, epoch)
		val_loss, results, truths = multimodal_evaluate(cfg, val_loader, model, loss_fn, epoch)
		val_wf = eval_meld(results, truths)
		test_loss, results, truths = multimodal_evaluate(cfg, test_loader, model, loss_fn, epoch, test=True)
		test_wf = eval_meld(results, truths, test=True)

		writer.add_scalar("Loss/train", train_loss, epoch)
		writer.add_scalar("Loss/val", val_loss, epoch)
		writer.add_scalar("WF/val", val_wf, epoch)
		writer.add_scalar("WF/test", test_wf, epoch)
		
		if best_val_wf < val_wf:
			best_val_wf = val_wf
			save_model(model, optimizer, cfg, model_name=cfg.train.vle_model_name)
		log.info(f'Consumed {(time.time() - ts)/3600} h for epoch {epoch} | val wf: {val_wf} | test wf: {test_wf}')

	
	# cfg.train.save_model and save_multimodal_model(model, optimizer, cfg, curr_time=strftime("%m-%d-%H-%M-%S"), device_id=device_id, trial_name=trial_name)


if __name__ == "__main__":
	os.environ['TOKENIZERS_PARALLELISM']='false'
	os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
	main()
