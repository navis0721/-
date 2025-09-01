import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from cross_models.cross_encoder import Encoder
from cross_models.cross_decoder import Decoder
from cross_models.attn import FullAttention, ProbAttention, AttentionLayer, TwoStageAttentionLayer
from cross_models.cross_embed import DSW_embedding

from math import ceil

class Crossformer(nn.Module):
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, attn='prob', num_experts=3, top_k=2, baseline = False, device=torch.device('cuda:0')):
        super(Crossformer, self).__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.attn = attn

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, attn, num_experts, top_k, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        # 建立可學習參數(standard distribution隨機抽樣)。//代表取整數
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, attn, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
    def forward(self, x_seq):
        # print("initial input: ", x_seq.shape)   # [batch_size(32), in_seq(48), columns(7)]
        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            # 應該是padding
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq) # DSW embedding
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)
        # print("x_seq after encoder: ", len(enc_out))  # 3, 每個encoder block的attn block數

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        # print("decoder input: ", len(dec_in))   # 32, batch_size
        predict_y = self.decoder(dec_in, enc_out)   # [batch_size(32), out_seq(24), columns(7)]

        return base + predict_y[:, :self.out_len, :]