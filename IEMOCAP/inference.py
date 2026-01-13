import glob
import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
import gc

from preprocessing import *
from utils import *
from dataset import *
from model import *

import matplotlib.pyplot as plt
import seaborn as sns
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=8, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='learning rate for training.')
    parser.add_argument('--batch_size', default=4, type=int, help='batch for training.')
    parser.add_argument('--seed', default=42, type=int, help='random seed fix')
    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def CELoss(pred_outs, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    fusion.eval()
    label_list = []
    pred_list = []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits,attn_nv,attn_cross,attn_reverse,embedding_output,text_emb,av_embedd = fusion(text_hidden, video_hidden, audio_hidden)
            
            """Calculation"""    
            
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()
            
            pred_list.extend(pred_label)
            label_list.extend(true_label)

    return pred_list, label_list

def plot_confusion_matrix(y_true, y_pred, labels, normalize=True, save_path="confusion_matrix.png"):
    if normalize:
        cm = confusion_matrix(y_true, y_pred, normalize='true')
        annot_format = ".2f"
    else:
        cm = confusion_matrix(y_true, y_pred)
        annot_format = "d"

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=annot_format, cmap='Blues',
                xticklabels=labels, yticklabels=labels,annot_kws={"size": 20})

    plt.xlabel('Predicted',fontsize=20)
    plt.ylabel('True',fontsize=20)
    plt.title('IEMOCAP Confusion Matrix',fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def print_incorrect_samples(y_true, y_pred, dataset_csv_path, emoList, save_path="incorrect_samples.txt"):
    df = pd.read_csv(dataset_csv_path)

    incorrect_indices = [i for i, (true, pred) in enumerate(zip(y_true, y_pred)) if true != pred]
    incorrect_count = len(incorrect_indices)

    with open(save_path, 'w', encoding='utf-8') as f:
        for idx in incorrect_indices:
            row = df.iloc[idx]
            true_label = emoList[y_true[idx]]
            pred_label = emoList[y_pred[idx]]
            text = (
                f"Utterance: {row['Utterance']}\n"
                f"Speaker: {row['Speaker']}\n"
                f"Dialogue_ID: {row['Dialogue_ID']}\n"
                f"Wav Path: {row['Wav_Path']}\n"
                f"Video Path: {row['Video_Path']}\n"
                f"True Label: {true_label}\n"
                f"Predicted Label: {pred_label}\n"
                f"{'-'*50}\n"
            )
            f.write(text)

        summary_text = f"Total Incorrect Samples: {incorrect_count}\n"
    print(f"Total Incorrect Samples: {incorrect_count}")

def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""

    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/IEMOCAP_full_release/'

    train_path = data_path + 'IEMOCAP_train.csv'
    dev_path = data_path + 'IEMOCAP_dev.csv'
    test_path = data_path + 'IEMOCAP_test.csv'

    test_dataset = iemocap_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    clsNum = len(test_dataset.emoList)
    init_config = Config()

    '''teacher model load'''
    model_t = Teacher_model(text_model, clsNum)
    model_t.load_state_dict(torch.load('./IEMOCAP/save_model/teacher.bin'))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    '''student model'''
    audio_s = Student_Audio(audio_model, clsNum, init_config)
    audio_s.load_state_dict(torch.load('./IEMOCAP/save_model/student_audio/total_student.bin'))
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()

    video_s = Student_Video(video_model, clsNum)
    video_s.load_state_dict(torch.load('./IEMOCAP/save_model/student_video/total_student.bin'))
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()

    '''fusion'''
    hidden_size, beta_shift, dropout_prob, num_head = 768, 2e-1, 0.2, 4
    fusion = ASF(clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion.load_state_dict(torch.load('./IEMOCAP/save_model/total_fusion.bin')) 
    for para in fusion.parameters():
        para.requires_grad = False
    fusion = fusion.cuda()
    fusion.eval()

    save_path = './IEMOCAP/figures/result/confusion_matrix_IEMOCAP.png'

    """Training Setting"""        
    test_pred_list, test_label_list = evaluation(model_t, audio_s, video_s, fusion, test_loader)
    print(classification_report(test_label_list, test_pred_list, target_names=test_dataset.emoList, digits=5))
    plot_confusion_matrix(test_label_list, test_pred_list, test_dataset.emoList, normalize=False, save_path=save_path)
    print_incorrect_samples(
        y_true=test_label_list,
        y_pred=test_pred_list,
        dataset_csv_path=test_path,
        emoList=test_dataset.emoList,
        save_path="./IEMOCAP/figures/result/incorrect_IEMOCAP.txt"
    )
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
