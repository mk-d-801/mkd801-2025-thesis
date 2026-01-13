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

from sklearn.metrics import precision_recall_fscore_support

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
from iemocap_kd import *
from triplet import *
from classbalanced import *

import matplotlib.pyplot as plt
import seaborn as sns
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=20, type=int, help='epoch for training.')
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
    loss_val = CBLoss(pred_outs,labels)
    return loss_val

def plot_attention(attn_map, title="Attention Map", save_path=None,file_name=None):
        plt.figure(figsize=(12, 10))
        sns.heatmap(attn_map.detach().cpu(), cmap="viridis")
        plt.title(title,fontsize=20)
        plt.xlabel("Key",fontsize=20)
        plt.ylabel("Query",fontsize=20)
        full_path = os.path.join(save_path, file_name)
        plt.savefig(full_path, bbox_inches='tight')
        plt.close()

def model_train(training_epochs, model_t, audio_s, video_s, fusion, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, scaler, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_dev_epoch , best_test_epoch = 0,0

    dev_losses = []
    test_losses = []
    dev_e_losses = []
    test_e_losses = []
    dev_tp_losses = []
    test_tp_losses = []

    dev_fscores = []
    test_fscores = []

    for epoch in tqdm(range(training_epochs)):
        fusion.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                """Prediction"""
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
                batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
                text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
                audio_hidden, audio_logits = audio_s(audio_inputs)
                video_hidden, video_logits = video_s(video_inputs)
                
                pred_logits,attn_nv,attn_cross,attn_reverse,embedding_output,text_emb,av_embedd = fusion(text_hidden, audio_hidden, video_hidden)

                loss_val = CELoss(pred_logits, batch_labels) + 0.1 * triplet(embedding_output,batch_labels)
            
            scaler.scale(loss_val).backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

        fusion.eval()   
        dev_pred_list, dev_label_list, dev_loss,dev_e_loss,dev_tp_loss = evaluation(model_t, audio_s, video_s, fusion, dev_dataloader)
        dev_losses.append(dev_loss)
        dev_e_losses.append(dev_e_loss)
        dev_tp_losses.append(dev_tp_loss)

        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        dev_fscores.append(dev_fbeta)
        print(f"Epoch{epoch+1} - dev_score : {dev_fbeta}")
        print(f"dev_loss : {dev_loss:.4f}, dev_loss(Emotion) : {dev_e_loss:.4f}, dev_loss(Triplet) : {dev_tp_loss:.4f}")

        test_pred_list, test_label_list, test_loss,test_e_loss,test_tp_loss = evaluation(model_t, audio_s, video_s, fusion, test_dataloader)
        test_losses.append(test_loss)
        test_e_losses.append(test_e_loss)
        test_tp_losses.append(test_tp_loss)

        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')                
        test_fscores.append(test_fbeta)
        print(f"Epoch{epoch+1} - test_score : {test_fbeta}")
        print(f"test_loss : {test_loss:.4f}, test_loss(Emotion) : {test_e_loss:.4f}, test_loss(Triplet) : {test_tp_loss:.4f}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_dev_epoch = epoch
            _SaveModel(fusion, save_path)
        
        if test_fbeta > best_test_fscore:
            best_test_fscore = test_fbeta
            best_test_epoch = epoch
    
    figure_save = './IEMOCAP/figures/fusion/lc'
    attn_save = 'IEMOCAP/figures/fusion/attention'
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_losses, label="Dev Loss")
    plt.plot(range(1, training_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("IEMOCAP fusion loss",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'IEMOCAP_fusion_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_e_losses, label="Dev Loss(Emotion)")
    plt.plot(range(1, training_epochs + 1), dev_tp_losses, label="Dev Loss(Triplet)")
    plt.plot(range(1, training_epochs + 1), dev_losses, label="Dev Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("IEMOCAP fusion loss (Dev)",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'IEMOCAP_fusion_dev_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), test_e_losses, label="Test Loss(Emotion)")
    plt.plot(range(1, training_epochs + 1), test_tp_losses, label="Test Loss(Triplet)")
    plt.plot(range(1, training_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("IEMOCAP fusion loss (Test)",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'IEMOCAP_fusion_test_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_fscores, label='Dev F-score')
    plt.plot(range(1, training_epochs + 1), test_fscores, label='Test F-score')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('F-score',fontsize=20)
    plt.title('IEMOCAP fusion f-score',fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'IEMOCAP_fusion_fscore.png'))
    plt.close()
    
    plot_attention(attn_nv,title="Non-Verval Self Attention Map",save_path=attn_save,file_name="Self-Attention.png")
    plot_attention(attn_cross,title="Cross Attention Map(Verval)",save_path=attn_save,file_name="Cross-Attention-VtoNV.png")
    plot_attention(attn_reverse,title="Cross Attention Map(Non-Verval)",save_path=attn_save,file_name="Cross-Attention-NVtoV.png")
    print(f"Best Dev F-score: {best_dev_fscore:.4f} at Epoch {best_dev_epoch+1}")
    print(f"Best Test F-score:{best_test_fscore:.4f} at Epoch {best_test_epoch+1}")

def evaluation(model_t, audio_s, video_s, fusion, dataloader):
    fusion.eval()
    label_list = []
    pred_list = []
    total_loss,total_loss_e,total_loss_tp = 0.0,0.0,0.0
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = batch_input_tokens.cuda(), attention_masks.cuda(), audio_inputs.cuda(), video_inputs.cuda(), batch_labels.cuda()
                
            text_hidden, test_logits = model_t(batch_input_tokens, attention_masks)
            audio_hidden, audio_logits = audio_s(audio_inputs)
            video_hidden, video_logits = video_s(video_inputs)
                
            pred_logits,attn_nv,attn_cross,attn_reverse,embedding_output,text_emb,av_embedd = fusion(text_hidden, audio_hidden, video_hidden)
            loss_e = CELoss(pred_logits,batch_labels)
            loss_tp = 0.1 * triplet(embedding_output,batch_labels)
            total_loss += (loss_e + loss_tp).item()
            total_loss_e += loss_e.item()
            total_loss_tp += loss_tp.item()
            """Calculation"""    
            
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)
    
    avg_loss = total_loss / len(dataloader)
    e_loss = total_loss_e / len(dataloader)
    tp_loss = total_loss_tp / len(dataloader)
    return pred_list, label_list, avg_loss,e_loss,tp_loss

def save_embeddings(model_t, audio_s, video_s, fusion, dataloader, save_file):
    fusion.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels = data
            batch_input_tokens = batch_input_tokens.cuda()
            attention_masks = attention_masks.cuda()
            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            batch_labels = batch_labels.cuda()

            text_hidden, _ = model_t(batch_input_tokens, attention_masks)
            audio_hidden, _ = audio_s(audio_inputs)
            video_hidden, _ = video_s(video_inputs)

            _,  _, _,_,embedding_output,_,_ = fusion(text_hidden, audio_hidden, video_hidden)

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
    torch.save(model.state_dict(), os.path.join(path, 'total_fusion.bin'))

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
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs)


    dev_dataset = iemocap_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    test_dataset = iemocap_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    save_path = os.path.join('./IEMOCAP/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
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
    fusion = fusion.cuda()
    fusion.eval()


    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    scaler = torch.cuda.amp.GradScaler()

    model_train(training_epochs, model_t, audio_s, video_s, fusion, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, scaler, save_path)
    save_embeddings(
    model_t, audio_s, video_s, fusion, 
    test_loader,
    "./IEMOCAP/figures/embedding/pt/fusion_embedding.pt"
)
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)