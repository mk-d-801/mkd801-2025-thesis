import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel , ElectraTokenizer, ElectraModel , DebertaV2TokenizerFast, DebertaV2Model
import gc

from preprocessing import *
from utils import *
from dataset import *
from model import *
from meld_kd import * 

import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Send.send import discord

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--learning_rate', default=5e-7, type=float, help='learning rate for training.')
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

def CELoss_ls(pred_outs, labels):
    loss = nn.CrossEntropyLoss(label_smoothing=0.02)
    loss_val = loss(pred_outs, labels)
    return loss_val

def CELoss(pred_outs, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_dev_epoch , best_test_epoch = 0,0

    dev_losses = []
    test_losses = []
    dev_e_losses = []
    test_e_losses = []
    dev_s_losses = []
    test_s_losses = []

    dev_e_fscores = []
    test_e_fscores = []
    dev_s_fscores = []
    test_s_fscores = []

    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels, batch_sublabels = data
            batch_input_tokens, batch_attention_masks, batch_labels, batch_sublabels = batch_input_tokens.cuda(),batch_attention_masks.cuda(), batch_labels.cuda(), batch_sublabels.cuda()
            last_hidden, pred_emotion, pred_sentiment = model(batch_input_tokens, batch_attention_masks)

            loss_e = CELoss(pred_emotion, batch_labels)
            loss_s = CELoss(pred_sentiment, batch_sublabels)
            loss_val = loss_e + loss_s
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()            
  
        model.eval()   
        dev_pred_list, dev_label_list, dev_subpred_list, dev_sublabel_list, dev_loss, dev_e_loss,dev_s_loss = evaluation(model, dev_dataloader)
        dev_losses.append(dev_loss)
        dev_e_losses.append(dev_e_loss)
        dev_s_losses.append(dev_s_loss)

        dev_pre_e, dev_rec_e, dev_fbeta_e, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        dev_pre_s, dev_rec_s, dev_fbeta_s, _ = precision_recall_fscore_support(dev_sublabel_list, dev_subpred_list, average='weighted')
        dev_e_fscores.append(dev_fbeta_e)
        dev_s_fscores.append(dev_fbeta_s)
        print(f"Epoch{epoch+1} - dev_score(Emotion) : {dev_fbeta_e}, dev_score(Sentiment) : {dev_fbeta_s}")
        print(f"dev_loss : {dev_loss:.4f} , dev_loss(Emotion) : {dev_e_loss:.4f}, dev_loss(Sentiment) : {dev_s_loss:.4f}")

        test_pred_list, test_label_list, test_subpred_list, test_sublabel_list ,test_loss, test_e_loss,test_s_loss = evaluation(model, test_dataloader)
        test_losses.append(test_loss)
        test_e_losses.append(test_e_loss)
        test_s_losses.append(test_s_loss)

        test_pre_e, test_rec_e, test_fbeta_e, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        test_pre_s, test_rec_s, test_fbeta_s, _ = precision_recall_fscore_support(test_sublabel_list, test_subpred_list, average='weighted')
        test_e_fscores.append(test_fbeta_e)
        test_s_fscores.append(test_fbeta_s)
        print(f"Epoch{epoch+1} - test_score(Emotion) : {test_fbeta_e}, test_score(Sentiment) : {test_fbeta_s}")
        print(f"test_loss : {test_loss:.4f}, test_loss(Emotion) : {test_e_loss:.4f}, test_loss(Sentiment) : {test_s_loss:.4f}")
        if dev_fbeta_e > best_dev_fscore:
            best_dev_fscore = dev_fbeta_e
            best_dev_epoch = epoch
            _SaveModel(model, save_path)

        if test_fbeta_e > best_test_fscore:
            best_test_fscore = test_fbeta_e
            best_test_epoch = epoch
    
    figure_save = './MELD/figures/teacher/'
    
    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_losses, label="Dev Loss")
    plt.plot(range(1, training_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("MELD text loss",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'MELD_text_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_e_losses, label="Dev Loss(Emotion)")
    plt.plot(range(1, training_epochs + 1), dev_s_losses, label="Dev Loss(Sentiment)")
    plt.plot(range(1, training_epochs + 1), dev_losses, label="Dev Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("MELD text loss (Dev)",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'MELD_text_dev_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), test_e_losses, label="Test Loss(Emotion)")
    plt.plot(range(1, training_epochs + 1), test_s_losses, label="Test Loss(Sentiment)")
    plt.plot(range(1, training_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("MELD text loss (Test)",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'MELD_text_test_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_e_fscores, label='Dev F-score')
    plt.plot(range(1, training_epochs + 1), test_e_fscores, label='Test F-score')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('F-score',fontsize=20)
    plt.title('MELD text f-score(Emotion)',fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'MELD_text_fscore_E.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_s_fscores, label='Dev F-score')
    plt.plot(range(1, training_epochs + 1), test_s_fscores, label='Test F-score')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('F-score',fontsize=20)
    plt.title('MELD text f-score(Sentiment)',fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'MELD_text_fscore_S.png'))
    plt.close()

    print(f"Best Dev F-score: {best_dev_fscore:.4f} at Epoch {best_dev_epoch+1}")
    print(f"Best Test F-score:{best_test_fscore:.4f} at Epoch {best_test_epoch+1}")

def evaluation(model, dataloader):
    model.eval()
    label_list, sublabel_list = [] , []
    pred_list, subpred_list  = [], []
    total_loss,total_loss_e,total_loss_s = 0.0,0.0,0.0
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_audio, batch_video, batch_labels, batch_sublabels = data
            batch_input_tokens, batch_attention_masks, batch_labels, batch_sublabels = batch_input_tokens.cuda(),batch_attention_masks.cuda(), batch_labels.cuda(), batch_sublabels.cuda()

            last_hidden, pred_emotion, pred_sentiment = model(batch_input_tokens, batch_attention_masks)
            
            loss_e = nn.CrossEntropyLoss()(pred_emotion, batch_labels)
            loss_s = nn.CrossEntropyLoss()(pred_sentiment, batch_sublabels)
            total_loss += (loss_e+ loss_s).item()
            total_loss_e += loss_e.item()
            total_loss_s += loss_s.item()
            """Calculation"""  
            
            pred_e = pred_emotion.argmax(1).detach().cpu().numpy()
            pred_s = pred_sentiment.argmax(1).detach().cpu().numpy()
            true_e = batch_labels.detach().cpu().numpy()
            true_s = batch_sublabels.detach().cpu().numpy()
            
            pred_list.extend(pred_e)
            subpred_list.extend(pred_s)
            label_list.extend(true_e)
            sublabel_list.extend(true_s)
    
    e_loss = total_loss_e / len(dataloader)
    s_loss = total_loss_s / len(dataloader)
    avg_loss = total_loss / len(dataloader)

    return pred_list, label_list, subpred_list, sublabel_list, avg_loss,e_loss,s_loss

def save_embeddings(model,dataloader, save_file):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            batch_input_tokens, attention_masks, audio_inputs, video_inputs, batch_labels, batch_sublabels = data
            batch_input_tokens = batch_input_tokens.cuda()
            attention_masks = attention_masks.cuda()
            audio_inputs = audio_inputs.cuda()
            video_inputs = video_inputs.cuda()
            batch_labels = batch_labels.cuda()

            embedding_output,_,_ = model(batch_input_tokens,attention_masks)

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
    torch.save(model.state_dict(), os.path.join(path, 'teacher.bin'))

def main(args):

    seed_everything(args.seed)
    """Dataset Loading"""

    text_model = "roberta-large"

    data_path = './dataset/MELD.Raw/'

    train_path = data_path + 'train_meld_emo.csv'
    dev_path = data_path + 'dev_meld_emo.csv'
    test_path = data_path + 'test_meld_emo.csv'

    train_dataset = meld_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=make_batchs)

    dev_dataset = meld_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    test_dataset = meld_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=make_batchs)

    save_path = os.path.join('./MELD/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    emo_clsNum = len(train_dataset.emoList)
    sent_clsNum = len(train_dataset.senList)
    model = Teacher_model(text_model, emo_clsNum, sent_clsNum)
    model = model.cuda()
    model.eval()

    """Training Setting"""        
    training_epochs = args.epochs
    save_term = int(training_epochs/5)
    max_grad_norm = 10
    lr = args.learning_rate
    num_training_steps = len(train_dataset)*training_epochs
    num_warmup_steps = len(train_dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr) # , eps=1e-06, weight_decay=0.01
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model_train(training_epochs, model, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, save_path)
    save_embeddings(
        model,test_loader,"./MELD/figures/embedding/pt/text_embedding.pt"
    )
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)
