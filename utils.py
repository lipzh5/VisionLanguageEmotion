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



# def load_multimodal_model(cfg, best_model_time, load_path='', device_id=0, from_ckpt=True):
    
#     if from_ckpt:
#         model = MultimodalFactory.get_model_instance(cfg) #VALTransformer(cfg)
#         if not load_path:
#             load_path = osp.join(cfg.train.save_model_path, 'temp', f'multimodal_model_{cfg.modalities}_{best_model_time}.pth')
#         ckpt = torch.load(load_path, map_location='cpu')
#         state_dict = ckpt['model']
#         model = model.cuda(device_id)
#         model.load_state_dict(state_dict)
#         return model
#     if not load_path:
#         load_path = osp.join(cfg.train.save_model_path, 'temp', f'multimodal_model_{cfg.modalities}_{best_model_time}_G{device_id}.pt')
#     model = torch.load(load_path)
#     return model

# def set_audio_encoder(cfg):
#     model_name = cfg.model.audio_encoder.model_name  # data2vec: used in TelME
#     if model_name == 'data2vec':
#         audio_encoder = Data2VecAudioModel.from_pretrained(cfg.model.audio_encoder.pretrained_path)
#     elif model_name == 'wav2vec':
#         audio_encoder = Wav2Vec2Model.from_pretrained(cfg.model.audio_encoder.pretrained_path_wav2vec)
    
#     # self.audio_encoder_trainable = cfg.train.audio_encoder_trainable
#     audio_enc_trainable = cfg.train.resnet_trainable
#     audio_encoder.train(audio_enc_trainable)
#     audio_encoder.requires_grad_(audio_enc_trainable)
#     return audio_encoder

def set_vision_encoder(cfg):
    use_webface_pretrain = cfg.model.vision_encoder.use_webface_pretrain
    print(f'Inception uses webface pretrain: {use_webface_pretrain} \n ******')
    vision_encoder = InceptionResnetV1(pretrained='casia-webface') if use_webface_pretrain else InceptionResnetV1()

    vis_enc_trainable = cfg.train.resnet_trainable
    vision_encoder.train(vis_enc_trainable)
    vision_encoder.requires_grad_(vis_enc_trainable)
    return vision_encoder


