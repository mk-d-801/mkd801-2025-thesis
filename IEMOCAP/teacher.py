import os
import pandas as pd
import numpy as np
import argparse
import random
from tqdm import tqdm
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

def CELoss(pred_outs, labels):
    loss = nn.CrossEntropyLoss()
    loss_val = loss(pred_outs, labels)
    return loss_val

def model_train(training_epochs, model, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, max_grad_norm, save_path):
    best_dev_fscore, best_test_fscore = 0, 0   
    best_dev_epoch , best_test_epoch = 0,0

    dev_losses = []
    test_losses = []

    dev_fscores = []
    test_fscores = []

    for epoch in tqdm(range(training_epochs)):
        model.train() 
        for i_batch, data in enumerate(train_dataloader):
            optimizer.zero_grad()

            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_labels = data
            batch_input_tokens, batch_attention_masks, batch_labels = batch_input_tokens.cuda(),batch_attention_masks.cuda(), batch_labels.cuda()
            last_hidden, pred_logits = model(batch_input_tokens, batch_attention_masks)

            loss_val = CELoss(pred_logits, batch_labels)
            
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()     

        model.eval()   
        dev_pred_list, dev_label_list, dev_loss = evaluation(model, dev_dataloader)
        dev_losses.append(dev_loss)
        dev_pre, dev_rec, dev_fbeta, _ = precision_recall_fscore_support(dev_label_list, dev_pred_list, average='weighted')
        dev_fscores.append(dev_fbeta)
        print(f"Epoch{epoch+1} - dev_score : {dev_fbeta}, dev_loss : {dev_loss:.4f}")

        test_pred_list, test_label_list, test_loss = evaluation(model, test_dataloader)
        test_losses.append(test_loss)
        test_pre, test_rec, test_fbeta, _ = precision_recall_fscore_support(test_label_list, test_pred_list, average='weighted')
        test_fscores.append(test_fbeta)
        print(f"Epoch{epoch+1} - test_score : {test_fbeta}, test_loss : {test_loss:.4f}")

        if dev_fbeta > best_dev_fscore:
            best_dev_fscore = dev_fbeta
            best_dev_epoch = epoch
            _SaveModel(model, save_path)
        
        if test_fbeta> best_test_fscore:
            best_test_fscore = test_fbeta
            best_test_epoch = epoch
    
    figure_save = './IEMOCAP/figures/teacher/'

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_losses, label="Dev Loss")
    plt.plot(range(1, training_epochs + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch",fontsize=20)
    plt.ylabel("Loss",fontsize=20)
    plt.title("IEMOCAP text loss",fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'IEMOCAP_text_loss.png'))
    plt.close()

    plt.figure(figsize=(8,6))
    plt.plot(range(1, training_epochs + 1), dev_fscores, label='Dev F-score')
    plt.plot(range(1, training_epochs + 1), test_fscores, label='Test F-score')
    plt.xlabel('Epoch',fontsize=20)
    plt.ylabel('F-score',fontsize=20)
    plt.title('IEMOCAP text f-score',fontsize=20)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_save, 'IEMOCAP_text_fscore.png'))
    plt.close()

    print(f"Best Dev F-score: {best_dev_fscore:.4f} at Epoch {best_dev_epoch+1}")
    print(f"Best Test F-score:{best_test_fscore:.4f} at Epoch {best_test_epoch+1}")

def evaluation(model, dataloader):
    model.eval()
    label_list = []
    pred_list = []
    total_loss = 0.0
    
    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):            
            """Prediction"""
            batch_input_tokens, batch_attention_masks, batch_labels = data
            batch_input_tokens, batch_attention_masks, batch_labels = batch_input_tokens.cuda(),batch_attention_masks.cuda(), batch_labels.cuda()

            last_hidden, pred_logits = model(batch_input_tokens, batch_attention_masks)
            loss = nn.CrossEntropyLoss()(pred_logits,batch_labels)
            total_loss += loss.item()
            """Calculation"""  
            
            pred_label = pred_logits.argmax(1).detach().cpu().numpy() 
            true_label = batch_labels.detach().cpu().numpy()

            pred_list.extend(pred_label)
            label_list.extend(true_label)
    
    avg_loss = total_loss / len(dataloader)
    return pred_list, label_list, avg_loss

def save_embeddings(model,dataloader, save_file):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for i_batch, data in enumerate(dataloader):
            batch_input_tokens, attention_masks, batch_labels = data
            batch_input_tokens = batch_input_tokens.cuda()
            attention_masks = attention_masks.cuda()
            batch_labels = batch_labels.cuda()

            embedding_output,_ = model(batch_input_tokens,attention_masks)

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

    data_path = './dataset/IEMOCAP_full_release/'

    train_path = data_path + 'IEMOCAP_train.csv'
    dev_path = data_path + 'IEMOCAP_dev.csv'
    test_path = data_path + 'IEMOCAP_test.csv'


    train_dataset = iemocap_dataset(preprocessing(train_path))
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=16, collate_fn=teacher_batchs)


    dev_dataset = iemocap_dataset(preprocessing(dev_path))
    dev_loader = DataLoader(dev_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=teacher_batchs)

    test_dataset = iemocap_dataset(preprocessing(test_path))
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=16, collate_fn=teacher_batchs)

    save_path = os.path.join('./IEMOCAP/save_model')
    
    print("###Save Path### ", save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    clsNum = len(train_dataset.emoList)
    model = Teacher_model(text_model, clsNum)
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    model_train(training_epochs, model, train_loader, dev_loader, test_loader, optimizer, scheduler, max_grad_norm, save_path)
    save_embeddings(
        model,test_loader,"./IEMOCAP/figures/embedding/pt/text_embedding.pt"
    )
    print("---------------Done--------------")

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    args = parse_args()
    main(args)