import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np

from math import sqrt

class FullAttention(nn.Module):
    '''
    The Attention operation
    '''
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        # print("attention: full")
        
    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)
        # print("Q, K: ", queries.shape, keys.shape)
        # print("V: ", values.shape)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # print("scores: ", scores.shape)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        # print("final attn: ", V.shape)
        
        return V.contiguous()
    
class ProbAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1):
        super(ProbAttention, self).__init__()
        # self.factor = factor
        self.factor = 1  #先固定，之後再調整(因為會跟crossformer的router數量搞混)
        self.scale = scale
        # self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        # print("attention: prob")

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape
        # print("Q, K shape: ", Q.shape, K.shape)
        # print("num of sample: ", sample_k)

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        # Q與隨機取樣後的K相乘
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparsity measurement(論文裡面那個公式)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # print(M.shape)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k
        # print("QK mat: ", Q_K.shape)

        return Q_K, M_top
        # return Q_K

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        # if not self.mask_flag:
        #     V_sum = V.sum(dim=-2)
        #     V_sum = V.mean(dim=-2)
        #     contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        # else: # use mask
        #     assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
        #     contex = V.cumsum(dim=-2)
        V_sum = V.sum(dim=-2)
        V_sum = V.mean(dim=-2)
        contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape  # [224, 4, 8, 64]

        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1) # nn.Softmax(dim=-1)(scores)
        tmp = torch.matmul(attn, V)
        # print("QK * V: ", tmp.shape)      # [224, 4, 3, 64]
        # 更新權重(上下文)
        # print("index: ", index.shape)   # [224, 4, 3]
        
        # 將挑出的[224, 4, 3, 64]的attn值更新[224, 4, 10, 64]context_in的特定位置(V均值)
        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)    
        # print("context_in: ", context_in.shape)    # [224, 4, 10, 64]
                   
        # if self.output_attention:
        #     attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
        #     attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
        #     return (context_in, attns)
        # else:
        #     return (context_in, None)
        return (context_in, None)

    def forward(self, queries, keys, values):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        # 決定取樣數 (key取U_part個計算KL divergence，query取前u個)
        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 
        # print("U_part, u: ", U_part, u)

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        # print("Q_K: ", scores_top.shape)    # index代表attention map中要更新的值
        # print("V: ", values.shape)

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context  (先建立一個大小為[224, 4, 8, 64]，並把值都初始化成加權平均的attn map)
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries  
        # (只把部分位置的值替換成[224, 4, 3, 64]中probsparse attn map中的值)
        context, attn = self._update_context(context, values, scores_top, index, L_Q)
        # print("context: ", context.shape)
        # tmp = context.transpose(2,1).contiguous()
        # print("context after transpose: ", tmp.shape)
        
        # return context.transpose(2,1).contiguous(), attn
        return context.transpose(2,1).contiguous()


class AttentionLayer(nn.Module):
    '''
    The Multi-head Self-Attention (MSA) Layer
    '''
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        # self.inner_attention = FullAttention(scale=None, attention_dropout = dropout)
        self.inner_attention = attention  # 依參數選擇要prob或full

        # 在這裡學習QKV矩陣
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        # print("attn in 2stage before attn: ", queries.shape, keys.shape, values.shape)

        # 做prob或full
        out = self.inner_attention(
            queries,
            keys,
            values,
        )
        # print("attn in 2stage: ", out.shape)

        out = out.view(B, L, -1)
        # print("attn in 2stage after view(B, L, -1): ", out.shape)

        return self.out_projection(out)

class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''
    def __init__(self, seg_num, factor, attn, d_model, n_heads, d_ff = None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.attn = attn
        Attn = ProbAttention if attn=='prob' else FullAttention
        self.time_attention = AttentionLayer(Attn(factor, dropout), d_model, n_heads)
        self.dim_sender = AttentionLayer(Attn(factor, dropout), d_model, n_heads)
        self.dim_receiver = AttentionLayer(Attn(factor, dropout), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))
        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                nn.GELU(),
                                nn.Linear(d_ff, d_model))

    def forward(self, x):
        #Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        # print("cross time stage before attn: ", x.shape)
        # print("batch: ", batch)
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        # print("time_in: ", time_in.shape)
        time_enc = self.time_attention(
            time_in, time_in, time_in
        )
        # print("time_in after self attention: ", time_enc.shape)
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)
        # print("cross time stage after attn: ", dim_in.shape)
        
        if self.attn == 'prob':
            # 改probsparse試試
            dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
            # print("cross dim stage before attn: ", dim_send.shape)  # [256, 7, 256]
            dim_buffer = self.dim_sender(dim_send, dim_send, dim_send)
            dim_enc = dim_send + self.dropout(dim_buffer)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)
            # print("cross dim stage after attn: ", dim_buffer.shape)  # [256, 7, 4, 64]
        else:
            #Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
            dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b = batch)
            # print("cross dim stage before attn: ", dim_send.shape)  # [batch*seq_num(32*8)), columns(7), d_model(256)]
            self.router = nn.Parameter(torch.randn(x.shape[2], self.router.shape[1], self.router.shape[2]))
            batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat = batch)
            # print("batch router: ", batch_router.shape)   # [batch*seq_num(32*8)), factor(10), d_model(256)]
            dim_buffer = self.dim_sender(batch_router, dim_send, dim_send)
            # print("dim_buffer: ", dim_buffer.shape)   # [batch*seq_num(32*8)), factor(10), d_model(256)]
            dim_receive = self.dim_receiver(dim_send, dim_buffer, dim_buffer)
            # print("dim receive: ", dim_receive.shape)
            dim_enc = dim_send + self.dropout(dim_receive)
            dim_enc = self.norm3(dim_enc)
            dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
            dim_enc = self.norm4(dim_enc)
            # print("cross dim stage after attn: ", dim_enc.shape)


        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b = batch)
        # print("final for 2stage attn: ", final_out.shape)

        return final_out
