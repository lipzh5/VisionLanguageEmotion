

EMOTIONS = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
EMOTION_LABELS = {em: i for i, em in enumerate(EMOTIONS)}

SPEAKER_LABELS = {
'chandler' : 0, 'monica' :1, 'phoebe': 2, 'rachel': 3, 'ross': 4, 'joey': 5,
}


LEARDING_ROLES = {
'chandler', 
'monica',
'phoebe',
'rachel',
'ross',
'joey',
}

NORMAL_MEAN = [0.5, 0.5, 0.5]
NORMAL_STD = [0.5, 0.5, 0.5]
RESNET_IMG_SIZE = 160  # InceptionResnetV1 

# MAX_DIA_LEN = 33  #(test集合中dia17的个数)

TEXT_MAX_UTT_LEN = 38  # MELD数据集中最长的utterance文本长度规定为38 其实最长有90
VISION_MAX_UTT_LEN = 100 # max frame list len in utterances
MAX_UTT_IMG_LEN = 174  # extracted by penny

AUDIO_IVALUE_MAX_LEN = 53861 #   400000   # 200000  # num_token: 624
# 200000: num_token 624
# 20000  num_token: 62
# 400000 # ref to TelME(num_token:1249)
# 53861  num_token 168 # median  # TODO adjust this value later

CONFIG = {}
# CONFIG['CLS'] = CLS  
# CONFIG['SEP'] = SEP
# CONFIG['mask_value'] = MASK
ATTN_MASK_FILL = -1e38 # -1e-9  #


