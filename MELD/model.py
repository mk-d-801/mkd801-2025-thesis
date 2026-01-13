import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd


from transformers import RobertaTokenizer, RobertaModel, TimesformerModel, Data2VecAudioModel
from meld_kd import *

class Teacher_model(nn.Module):
    def __init__(self, text_model, emo_clsNum, sent_clsNum):
        super(Teacher_model, self).__init__()
        
        """Text Model"""
        tmodel_path = text_model
        if text_model == 'roberta-large':
            self.text_model = RobertaModel.from_pretrained(tmodel_path)
            tokenizer = RobertaTokenizer.from_pretrained(tmodel_path)
            self.speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
            self.speaker_tokens_dict = {'additional_special_tokens': self.speaker_list}
            tokenizer.add_special_tokens(self.speaker_tokens_dict)
            
        self.text_model.resize_token_embeddings(len(tokenizer))
        self.text_hiddenDim = self.text_model.config.hidden_size
        
        """Logit"""
        self.W = nn.Linear(self.text_hiddenDim, 768)

        self.classifier_e = nn.Linear(768, emo_clsNum)
        self.classifier_s = nn.Linear(768, sent_clsNum)

    def forward(self, batch_input_tokens, attention_masks):

        batch_context_output = self.text_model(batch_input_tokens, attention_masks).last_hidden_state[:,-1,:]
        
        batch_last_hidden = self.W(batch_context_output)

        context_logit_e = self.classifier_e(batch_last_hidden)
        context_logit_s = self.classifier_s(batch_last_hidden)
        
        return batch_last_hidden, context_logit_e, context_logit_s

class Student_Audio(nn.Module):
    def __init__(self, audio_model, emo_clsNum, sent_clsNum, init_config):
        super(Student_Audio, self).__init__()
        
        """Model Setting"""
        amodel_path = audio_model
        if audio_model == "facebook/data2vec-audio-base-960h":
            
            self.model = Data2VecAudioModel.from_pretrained(amodel_path)
            self.model.config.update(init_config.__dict__)

        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W_e = nn.Linear(self.hiddenDim, emo_clsNum)
        self.W_s = nn.Linear(self.hiddenDim, sent_clsNum)

    def forward(self, batch_input):

        batch_audio_output = self.model(batch_input).last_hidden_state[:,0,:] 

        audio_logit_e = self.W_e(batch_audio_output)
        audio_logit_s = self.W_s(batch_audio_output)
        
        return batch_audio_output, audio_logit_e, audio_logit_s


class Student_Video(nn.Module):
    def __init__(self, video_model, emo_clsNum, sent_clsNum):
        super(Student_Video, self).__init__()
        
        vmodel_path = video_model
        if video_model == "facebook/timesformer-base-finetuned-k400":

            self.model = TimesformerModel.from_pretrained(vmodel_path)

        self.hiddenDim = self.model.config.hidden_size
            
        """score"""
        self.W_e = nn.Linear(self.hiddenDim, emo_clsNum)
        self.W_s = nn.Linear(self.hiddenDim, sent_clsNum)

    def forward(self, batch_input):

        batch_video_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 768)

        video_logit_e = self.W_e(batch_video_output) # (batch, clsNum)
        video_logit_s = self.W_s(batch_video_output)
        
        return batch_video_output, video_logit_e, video_logit_s

class ASF(nn.Module):
    def __init__(self, emo_clsNum, sent_clsNum, hidden_size, beta_shift, dropout_prob, num_head):
        super(ASF, self).__init__()

        self.TEXT_DIM = 768
        self.VISUAL_DIM = 768
        self.ACOUSTIC_DIM = 768

        self.multihead_attn = nn.MultiheadAttention(self.VISUAL_DIM + self.ACOUSTIC_DIM, num_head)
        self.cross_attn = nn.MultiheadAttention(self.TEXT_DIM, 12)
        self.cross_ln = nn.LayerNorm(self.TEXT_DIM)
 
        self.proj_av_to_text = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM, self.TEXT_DIM)       
        self.W_hav = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.ACOUSTIC_DIM)
        self.dropout = nn.Dropout(dropout_prob)

        
        """Logit"""
        self.W_e = nn.Linear(self.TEXT_DIM, emo_clsNum)
        self.W_s = nn.Linear(self.TEXT_DIM, sent_clsNum)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        nv_embedd = torch.cat((visual, acoustic), dim=-1)
        text_emb = text_embedding

        new_nv,attn_nv = self.multihead_attn(nv_embedd, nv_embedd, nv_embedd)
        new_nv = new_nv + nv_embedd
        av_embedd = self.dropout(self.AV_LayerNorm(new_nv))
        

        av_proj = self.proj_av_to_text(av_embedd)

        cross_out,attn_cross = self.cross_attn(text_emb, av_proj, av_proj)
        cross_reverse,attn_reverse = self.cross_attn(av_proj, text_emb, text_emb)

        cross_out = self.cross_ln(cross_out + cross_reverse)

        weight_av = F.relu(self.W_hav(torch.cat((av_embedd, cross_out), dim=-1)))

        h_m = weight_av * self.W_av(av_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        logits_e = self.W_e(embedding_output)
        logits_s = self.W_s(embedding_output)
        return logits_e, logits_s,attn_nv,attn_cross,attn_reverse,embedding_output,text_emb,av_embedd