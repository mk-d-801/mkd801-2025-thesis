import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoProcessor, Data2VecAudioModel
from transformers import AutoImageProcessor, TimesformerModel
import gc

from preprocessing import *
from utils import *
from dataset import *
from model import *
from iemocap_kd import *

import matplotlib.pyplot as plt

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

def CE_Loss(pred_outs, logit_t, hidden_s, hidden_t, labels):
    ori_loss = nn.CrossEntropyLoss()
    ori_loss = ori_loss(pred_outs, labels)
    logit_loss = Logit_Loss().cuda()
    logit_loss = logit_loss(pred_outs, logit_t)
    feature_loss = Feature_Loss().cuda()
    feature_loss = feature_loss(hidden_s, hidden_t)

    loss_val = ori_loss + 0.1*logit_loss + feature_loss 
    return loss_val

def model_train(student_type, training_epochs, model_t, model_s, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_dev_epoch , best_test_epoch = 0,0

    dev_losses = []
    test_losses = []

    dev_fscores = []
    test_fscores = []

    model_t.eval()
    for epoch in tqdm(range(training_epochs)):
        model_s.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                """Prediction"""
                if student_type == "audio":
                    batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
                    batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_audio.cuda(), batch_labels.cuda()
                    hidden_s, logit_s = model_s(batch_audio)
                    hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)
                else:
                    batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = data
                    batch_input_tokens, batch_attention_masks, batch_video, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(), batch_video.cuda(), batch_labels.cuda()
                    hidden_s, logit_s = model_s(batch_video)
                    hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)
                loss_val = CE_Loss(logit_s, logit_t, hidden_s, hidden_t, batch_labels)
                  
            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(model_s.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        model_s.eval()   
        dev_pred_list, dev_label_list, dev_loss = evaluation(student_type, model_s, model_t,dev_dataloader)
        dev_losses.append(dev_loss)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        dev_fscores.append(dev_fbeta)
        print(f"Epoch{epoch+1} - dev_score : {dev_fbeta}, dev_loss : {dev_loss:.4f}")

        test_pred_list, test_label_list, test_loss = evaluation(student_type,model_s,model_t,test_dataloader)
        test_losses.append(test_loss)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        test_fscores.append(test_fbeta)
        print(f"Epoch{epoch+1} - test_score : {test_fbeta}, test_loss : {test_loss:.4f}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_dev_epoch = epoch
            _SaveModel(model_s, save_path)
        
        if test_fbeta > best_test_fscore:
            best_test_fscore = test_fbeta
            best_test_epoch = epoch

    figure_save = f'./IEMOCAP/figures/student/{student_type}/'

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_losses, label="Dev Loss")
    plt.plot(range(1, training_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title(f'IEMOCAP {student_type} loss',fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    loss_metrics_filename = f'IEMOCAP_{student_type}_loss.png'
    plt.savefig(os.path.join(figure_save, loss_metrics_filename))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_fscores, label='Dev F-score')
    plt.plot(range(1, training_epochs + 1), test_fscores, label='Test F-score')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('F-score',fontsize=20)
    plt.title(f'IEMOCAP {student_type} f-score',fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    f_metrics_filename = f'IEMOCAP_{student_type}_fscore.png'
    plt.savefig(os.path.join(figure_save, f_metrics_filename))
    plt.close()

    print(f"Best Dev F-score: {best_dev_fscore:.4f} at Epoch {best_dev_epoch+1}")
    print(f"Best Test F-score:{best_test_fscore:.4f} at Epoch {best_test_epoch+1}")


def evaluation(student_type, model_s, model_t,dataloader):
    model_s.eval()
    model_t.eval()
    label_list = []
    pred_list = []
    total_loss = 0.0
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            if student_type == "audio":
                batch_input_tokens, batch_attention_masks, batch_audio, batch_labels = data
                batch_input_tokens, batch_attention_masks,batch_audio, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(),batch_audio.cuda(), batch_labels.cuda()
                hidden_s, logit_s = model_s(batch_audio)
                hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)

            else:
                batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels = data
                batch_input_tokens, batch_attention_masks,batch_video, batch_labels = batch_input_tokens.cuda(), batch_attention_masks.cuda(),batch_video.cuda(), batch_labels.cuda()
                hidden_s, logit_s = model_s(batch_video)
                hidden_t, logit_t = model_t(batch_input_tokens, batch_attention_masks)
            
            loss = CE_Loss(logit_s,logit_t,hidden_s,hidden_t,batch_labels)
            total_loss += loss.item()

            """Calculation"""    
            
            pred_label = logit_s.argmax(1).detach().cpu().numpy()
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)
    
    avg_loss = total_loss / len(dataloader)
    return pred_list, label_list , avg_loss

def save_embeddings(student_type,model,dataloader, save_file):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            if student_type == "audio":
                batch_input_tokens, attention_masks, audio_inputs, batch_labels = data
                audio_inputs = audio_inputs.cuda()
                embedding_output,_ = model(audio_inputs)
            else:
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
                video_inputs = video_inputs.cuda()
                embedding_output,_ = model(video_inputs)

            embeddings.append(embedding_output.detach().cpu())
            labels.append(batch_labels.detach().cpu())

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    torch.save(
        {"embeddings": embeddings, "labels": labels},
        save_file
    )

    print(f"[Embedding Saved] -> {save_file}")

def _SaveModel(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), os.path.join(path, 'total_student.bin'))

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


    train_dataset = iemocap_dataset(preprocessing(train_path))
    audio_trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=audio_batchs)
    visual_trainloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs)
    
    dev_dataset = iemocap_dataset(preprocessing(dev_path))
    audio_devloader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=audio_batchs)
    visual_devloader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    test_dataset = iemocap_dataset(preprocessing(test_path))
    audio_testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=audio_batchs)
    visual_testloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    save_audio = os.path.join('./IEMOCAP/save_model', "student_audio")
    save_video = os.path.join('./IEMOCAP/save_model', "student_video")

    print("###Save Path### ", save_audio)
    if not os.path.exists(save_audio):
        os.makedirs(save_audio)
    
    print("###Save Path### ", save_video)
    if not os.path.exists(save_video):
        os.makedirs(save_video)
  
    clsNum = len(train_dataset.emoList)
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
    audio_s = audio_s.cuda()
    audio_s.eval()

    video_s = Student_Video(video_model, clsNum)
    video_s = video_s.cuda()
    video_s.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer_audio = torch.optim.AdamW(audio_s.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    optimizer_video = torch.optim.AdamW(video_s.parameters(), lr=lr)
    scheduler_audio = get_linear_schedule_with_warmup(optimizer_audio, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scheduler_video = get_linear_schedule_with_warmup(optimizer_video, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train("audio", training_epochs, model_t, audio_s, audio_trainloader, audio_devloader, audio_testloader, optimizer_audio, scheduler_audio, max_grad_norm, scaler, save_audio)
    model_train("visual", training_epochs, model_t, video_s, visual_trainloader, visual_devloader, visual_testloader, optimizer_video, scheduler_video, max_grad_norm, scaler, save_video)
    save_embeddings(
        "audio",audio_s,audio_testloader,"./IEMOCAP/figures/embedding/pt/audio_embedding.pt"
    )
    save_embeddings(
        "visual",video_s,visual_testloader,"./IEMOCAP/figures/embedding/pt/visual_embedding.pt"
    )
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)