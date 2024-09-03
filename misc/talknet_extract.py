# -*- coding:utf8 -*-
import cv2
import os,code
import shutil
import subprocess
import os.path as osp


# input_file = '\"E:\\EMNLP2020\\MELD.Raw\\dev_splits\\dia0_utt0.mp4"'
# output_dir = '\"E:\\EMNLP2020\\MELD.Raw\\dev_splits\\'



split_type = 'test'
source_path = f'/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/{split_type}'
# target_path = f'/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/openfacefeat_{split_type}'

data_root = '/home/penny/pycharmprojects/common/data/meld'
import pandas as pd
anno_csv_path = osp.join(data_root, f'{split_type}_sent_emo.csv')
anno_data = pd.read_csv(anno_csv_path)
# all_video_paths = []
# command = '/home/penny/pycharmprojects/OpenFaceTool/build/bin/FaceLandmarkImg -fdir ' + "/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/val/dia0_utt0/pyframes"  \
#     + ' -out_dir ' + "/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/val/dia0_utt0/landmarks/"
# subprocess.call(["/home/penny/pycharmprojects/OpenFaceTool/build/bin/FeatureExtraction", 
# "-fdir", "/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/val/dia0_utt0/pyframes", 
# "-out_dir", "/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/val/dia0_utt0/openface_feat/"])
# subprocess.call(["/home/penny/pycharmprojects/OpenFaceTool/build/bin/FeatureExtraction", 
# "-fdir", "/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/openfacefeat_val/dia49_utt5/dia49_utt5_aligned", 
# "-out_dir", "/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/openfacefeat_val/dia49_utt5/aligned_feat/"])
# raise ValueError('Penny stops here!!!')


def extract_meld_raw():
    import pandas as pd
    anno_csv_path = osp.join(data_root, f'{split_type}_sent_emo.csv')
    anno_data = pd.read_csv(anno_csv_path)
    source_path = f'/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/{split_type}/'
    target_path = f'/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/openfacefeat_{split_type}/'

    for row in anno_data.iterrows():
        meta = row[1]
        utt_name = f'dia{meta["Dialogue_ID"]}_utt{meta["Utterance_ID"]}'
        input_video_name = utt_name
        if split_type == 'test':
            try_video_name = f'final_videos_test{utt_name}'
            try_video_path = osp.join(source_path, f'{try_video_name}.mp4')
            if osp.exists(try_video_path):
                input_video_name = try_video_name

       
        # command = 'python ' + input_file  + ' -out_dir ' + output_dir
        os.chdir("/home/penny/pycharmprojects/talknet")
        command = f"python /home/penny/pycharmprojects/talknet/demoTalkNet.py --videoName {input_video_name} --videoFolder {source_path}"
        os.system(command)
        # subprocess.call(["python", "/home/penny/pycharmprojects/talknet/demoTalkNet.py", "--videoName", input_video_name, "--videoDir", source_path])
        print(f'video name: {input_video_name} \n****')
        # break
      

def move_file():
    import os
    import shutil
    split_type = 'test'
    source_path = f'/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/openfacefeat_{split_type}/'
    # target_path = f'/home/penny/pycharmprojects/common/data/meld/raw/MELD.Raw/openfacefeat_{split_type}_/'
    target_path = f'/home/penny/pycharmprojects/common/data/meld/preprocessed_data/vision/openfacefeat_masked_{split_type}/'
    files = os.listdir(source_path)
    print(f'len files: {len(files)}, {files[0]} \n *****')
    for dir_name in files:
        src = osp.join(source_path, dir_name, f'final_videos_test{dir_name}_aligned')
        if not osp.exists(src):
            src = osp.join(source_path, dir_name, f'{dir_name}_aligned_masked')
            # src = osp.join(source_path, dir_name, f'{dir_name}_aligned')
        dst = osp.join(target_path, dir_name)
        try:
            shutil.copytree(src, dst)
        except Exception as e:
            print(str(e))
            print(f'move file failed: {src} \n *****')
    pass
         

if __name__ == "__main__":
    # extract_from_aligned_imgs()
    # move_file()
    extract_meld_raw()

    pass