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
from transformers import RobertaTokenizer, RobertaModel , ElectraTokenizer, ElectraModel , DebertaV2TokenizerFast, DebertaV2Model
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
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
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

def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    label_list, sublabel_list = [], []
    pred_list, subpred_list = [], []
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels, batch_sublabels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels, batch_sublabels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda(), batch_sublabels.cuda()
                
            text_hidden, text_logits_e, text_logits_s = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits_e, audio_logits_s  = audio_s(audio_inputs)
            video_hidden, video_logits_e, video_logits_s  = video_s(video_inputs)
                
            pred_logits_e, pred_logits_s,attn_nv,attn_cross,attn_reverse,embedding_output,text_emb,av_embedd = fusion(text_hidden, audio_hidden, video_hidden)
            
            """Calculation"""    

            pred_label_e = pred_logits_e.argmax(1).detach().cpu().numpy() 
            true_label_e = batch_labels.detach().cpu().numpy()

            pred_label_s = pred_logits_s.argmax(1).detach().cpu().numpy() 
            true_label_s = batch_sublabels.detach().cpu().numpy()
            
            pred_list.extend(pred_label_e)
            subpred_list.extend(pred_label_s)
            label_list.extend(true_label_e)
            sublabel_list.extend(true_label_s)

    return pred_list, label_list, subpred_list, sublabel_list

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
    plt.title('MELD Confusion Matrix',fontsize=20)
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
                f"Sr No: {row['Sr No.']}\n"
                f"Speaker: {row['Speaker']}\n"
                f"Utterance: {row['Utterance']}\n"
                f"Video Path: {row['Video_Path']}\n"
                f"True Label: {true_label}\n"
                f"Predicted Label: {pred_label}\n"
                f"{'-'*50}\n"
            )
            f.write(text)
            summary_text = f"Total Incorrect Samples: {incorrect_count}\n"
    print(f"Total Incorrect Samples: {incorrect_count}")

def plot_confusion_matrix_per_speaker(y_true, y_pred, dataset_csv_path, labels, normalize=True, save_dir="./MELD/figures/result/",type = "None"):

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(dataset_csv_path)

    target_speakers = ['Rachel', 'Monica', 'Phoebe', 'Ross', 'Chandler', 'Joey']

    for speaker in target_speakers:
        speaker_indices = df.index[df['Speaker'] == speaker].tolist()
        if not speaker_indices:
            continue

        y_true_sp = [y_true[i] for i in speaker_indices if i < len(y_true)]
        y_pred_sp = [y_pred[i] for i in speaker_indices if i < len(y_pred)]

        if len(set(y_true_sp)) <= 1:
            print(f"Skipping {speaker} (only one class present).")
            continue

        if normalize:
            cm = confusion_matrix(y_true_sp, y_pred_sp, normalize='true')
            annot_format = ".2f"
        else:
            cm = confusion_matrix(y_true_sp, y_pred_sp)
            annot_format = "d"

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt=annot_format, cmap='Blues',
                    xticklabels=labels, yticklabels=labels,annot_kws={"size": 20})
        plt.xlabel("Predicted",fontsize=20)
        plt.ylabel("True",fontsize=20)
        plt.title(f"Confusion Matrix ({speaker})",fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        save_path = os.path.join(save_dir, f"confusion_matrix_{speaker}_{type}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def print_classification_report_per_speaker(y_true, y_pred, dataset_csv_path, emoList):
    df = pd.read_csv(dataset_csv_path)
    target_speakers = ['Rachel', 'Monica', 'Phoebe', 'Ross', 'Chandler', 'Joey']

    f1_matrix = pd.DataFrame(index=emoList + ['W-F1', 'M-F1', 'Acc'], columns=target_speakers)

    for speaker in target_speakers:
        speaker_indices = df.index[df['Speaker'] == speaker].tolist()
        if not speaker_indices:
            continue

        y_true_sp = [y_true[i] for i in speaker_indices if i < len(y_true)]
        y_pred_sp = [y_pred[i] for i in speaker_indices if i < len(y_pred)]

        report = classification_report(
            y_true_sp, y_pred_sp, target_names=emoList, output_dict=True, zero_division=0
        )

        for emo in emoList:
            f1_matrix.at[emo, speaker] = report.get(emo, {}).get('f1-score', 0.0)

        f1_matrix.at['W-F1', speaker] = report['weighted avg']['f1-score']
        f1_matrix.at['M-F1', speaker] = report['macro avg']['f1-score']
        f1_matrix.at['Acc', speaker]  = report['accuracy']

    f1_matrix = f1_matrix.fillna(0.0).round(5)
    print(f1_matrix)



def main(args):
    seed_everything(args.seed)
    @dataclass
    class Config():
        mask_time_length: int = 3
    """Dataset Loading"""
    
    text_model = "roberta-large"
    audio_model = "facebook/data2vec-audio-base-960h"
    video_model = "facebook/timesformer-base-finetuned-k400"

    data_path = './dataset/MELD.Raw/'

    train_path = data_path + 'train_meld_emo.csv'
    dev_path = data_path + 'dev_meld_emo.csv'
    test_path = data_path + 'test_meld_emo.csv'

    test_dataset = meld_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)
        
    emo_clsNum = len(test_dataset.emoList)
    sent_clsNum = len(test_dataset.senList)
    init_config = Config()

    '''teacher model load'''
    model_t = Teacher_model(text_model, emo_clsNum, sent_clsNum)
    model_t.load_state_dict(torch.load('./MELD/save_model/teacher.bin'))
    for para in model_t.parameters():
        para.requires_grad = False
    model_t = model_t.cuda()
    model_t.eval()

    '''student model'''
    audio_s = Student_Audio(audio_model, emo_clsNum, sent_clsNum, init_config)
    audio_s.load_state_dict(torch.load('./MELD/save_model/student_audio/total_student.bin')) 
    for para in audio_s.parameters():
        para.requires_grad = False
    audio_s = audio_s.cuda()
    audio_s.eval()

    video_s = Student_Video(video_model, emo_clsNum, sent_clsNum)
    video_s.load_state_dict(torch.load('./MELD/save_model/student_video/total_student.bin')) 
    for para in video_s.parameters():
        para.requires_grad = False
    video_s = video_s.cuda()
    video_s.eval()

    '''fusion'''
    hidden_size, beta_shift, dropout_prob, num_head = 768, 1e-1, 0.2, 3
    fusion = ASF(emo_clsNum, sent_clsNum, hidden_size, beta_shift, dropout_prob, num_head)
    fusion.load_state_dict(torch.load('./MELD/save_model/total_fusion.bin')) 
    for para in fusion.parameters():
        para.requires_grad = False
    fusion = fusion.cuda()
    fusion.eval()

    save_path = './MELD/figures/result/confusion_matrix_MELD_E.png'
    save_path_2 = './MELD/figures/result/confusion_matrix_MELD_S.png'

    """Training Setting"""        
    test_pred_list, test_label_list, test_subpred_list, test_sublabel_list = evaluation(model_t, audio_s, video_s, fusion, test_loader)
    print(classification_report(test_label_list, test_pred_list, target_names=test_dataset.emoList, digits=5))
    print(classification_report(test_sublabel_list, test_subpred_list, target_names=test_dataset.senList, digits=5 ))
    
    print_classification_report_per_speaker(
    y_true=test_label_list,
    y_pred=test_pred_list,
    dataset_csv_path=test_path,
    emoList=test_dataset.emoList
)
    print_classification_report_per_speaker(
    y_true=test_sublabel_list,
    y_pred=test_subpred_list,
    dataset_csv_path=test_path,
    emoList=test_dataset.senList
)
    
    plot_confusion_matrix(test_label_list, test_pred_list, test_dataset.emoList, normalize=False, save_path=save_path)
    plot_confusion_matrix(test_sublabel_list, test_subpred_list, test_dataset.senList, normalize=False, save_path=save_path_2)

    plot_confusion_matrix_per_speaker(
    y_true=test_label_list,
    y_pred=test_pred_list,
    dataset_csv_path=test_path,
    labels=test_dataset.emoList,
    normalize=False,
    save_dir="./MELD/figures/result/emotion",
    type='E'
)
    
    plot_confusion_matrix_per_speaker(
    y_true=test_sublabel_list,
    y_pred=test_subpred_list,
    dataset_csv_path=test_path,
    labels=test_dataset.senList,
    normalize=False,
    save_dir="./MELD/figures/result/sentiment",
    type='S'
)
    
    print_incorrect_samples(
    y_true=test_label_list,
    y_pred=test_pred_list,
    dataset_csv_path=test_path,
    emoList=test_dataset.emoList,
    save_path="./MELD/figures/result/incorrect_MELD_E.txt"
    )
    print_incorrect_samples(
    y_true=test_sublabel_list,
    y_pred=test_subpred_list,
    dataset_csv_path=test_path,
    emoList=test_dataset.senList,
    save_path="./MELD/figures/result/incorrect_MELD_S.txt"
    )

    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
