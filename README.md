# Vision-Lanugage to Emotion Model for Affective Human-Robot Interaction

## Introduction
This is the official implementation of the ML part of our paper **UGotMe: An Embodied System for Affective Human-Robot Interaction**

## Getting Started
### 1. Clone the code the prepare the environment
```
git clone git@github.com:lipzh5/VisionLanguageEmotion.git
cd VisionLanguageEmotion
# create env using conda for CUDA 12.1
conda create -n vl2e python=3.8 
conda activate vl2e
pip install -r requirements.txt

```
### 2. Run model training with your own hyperparameters
```
python main.py train.batch_size=8 train.accumulation_steps=2
```