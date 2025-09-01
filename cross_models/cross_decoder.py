import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FullAttention, ProbAttention, AttentionLayer, TwoStageAttentionLayer

class DecoderLayer(nn.Module):
    '''
    The decoder layer of Crossformer, each layer will make a prediction at its scale
    '''
    def __init__(self, seg_len, attn, d_model, n_heads, d_ff=None, dropout=0.1, out_seg_num = 10, factor = 10):
        super(DecoderLayer, self).__init__()
        self.self_attention = TwoStageAttentionLayer(out_seg_num, factor, attn, d_model, n_heads, \
                                d_ff, dropout) 
        # Attn = ProbAttention if attn=='prob' else FullAttention   
        self.cross_attention = AttentionLayer(FullAttention(factor, dropout), d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                nn.GELU(),
                                nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)
        self.attn = attn

    def forward(self, x, cross):
        '''
        x: the output of last decoder layer
        cross: the output of the corresponding encoder layer
        '''

        batch = x.shape[0]
        # print("x before attention: ", x.shape)
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')
        
        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        # print("self attention Q, K, V: ", x.shape, cross.shape, cross.shape)
        tmp = self.cross_attention(
            x, cross, cross,
        )
        # print("after self attention: ", tmp.shape)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x+y)
        # print("decoder output: ", dec_output.shape)
        
        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b = batch)
        # print("decoder output after rearrange: ", dec_output.shape)
        layer_predict = self.linear_pred(dec_output)
        # print("layer predict: ", layer_predict.shape)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')
        # print("layer predict after rearrange: ", layer_predict.shape)

        return dec_output, layer_predict

class Decoder(nn.Module):
    '''
    The decoder of Crossformer, making the final prediction by adding up predictions at each scale
    '''
    def __init__(self, seg_len, d_layers, attn, d_model, n_heads, d_ff, dropout,\
                router=False, out_seg_num = 10, factor=10):
        super(Decoder, self).__init__()

        self.router = router
        self.decode_layers = nn.ModuleList()
        self.attn = attn
        for i in range(d_layers):
            # print("---decoder---")
            self.decode_layers.append(DecoderLayer(seg_len, attn, d_model, n_heads, d_ff, dropout, \
                                        out_seg_num, factor))

    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x,  cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        
        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d = ts_d)
        # print("final predict: ", final_predict.shape)

        return final_predict

