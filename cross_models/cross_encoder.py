import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from cross_models.attn import FullAttention, ProbAttention, AttentionLayer, TwoStageAttentionLayer
from math import ceil

from torch import Tensor, nn
# from zeta.nn.modules.feedforward import FeedForward

# , MultiQueryAttention



class SwitchGate(nn.Module):
    """
    SwitchGate module for MoE (Mixture of Experts) model.

    Args:
        dim (int): Input dimension.
        num_experts (int): Number of experts.
        capacity_factor (float, optional): Capacity factor for sparsity. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """

    def __init__(
        self,
        dim,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.0,
        epsilon: float = 1e-6,
        *args,
        **kwargs,
    ):
        super().__init__()
        # 256 (256個特徵)
        self.dim = dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor # 確定每個token需要幾個專家
        self.epsilon = epsilon
        self.w_gate = nn.Linear(dim, num_experts)

    def forward(self, x: Tensor, use_aux_loss=False):
        """
        Forward pass of the SwitchGate module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Gate scores.
        """
        # Compute gate scores
        # torch.Size([32, 7, 8, 3])
        gate_scores = F.softmax(self.w_gate(x), dim=-1)
        # Determine the top-1 expert for each token
        capacity = int(self.capacity_factor * x.size(0))
        capacity = int(3)
        
        # 皆為 torch.Size([32, 7, 8, 2]) 一個是分數 一個是這些分數的位置
        top_k_scores, top_k_indices = gate_scores.topk(self.top_k, dim=-1)
        # Mask to enforce sparsity
        mask = torch.zeros_like(gate_scores).scatter_(
            1, top_k_indices, 1
        )
        # Combine gating scores with the mask
        masked_gate_scores = gate_scores * mask

        # Denominators
        # epsilon 避免值等於0，所以加上一個小小小數字
        denominators = (
            masked_gate_scores.sum(0, keepdim=True) + self.epsilon
        )

        # Norm gate scores to sum to the capacity
        # 讓每個專家的負載量一樣，不要一黨獨大
        gate_scores = (masked_gate_scores / denominators) * capacity
        # 32 8 3
        # 8 3
        # 32 8
        
        if use_aux_loss:
            # load:  torch.Size([7, 8, 3])
            load = gate_scores.sum(0)  # Sum over all examples
            # importance:  torch.Size([32, 7, 8])
            importance = gate_scores.sum(-1)  # Sum over all experts
            print("gate_scores: ", gate_scores.shape)
            print("load: ", load.shape)
            print("importance: ", importance.shape)
            # Aux loss is mean suqared difference between load and importance
            loss = ((load - importance) ** 2).mean()

            return gate_scores, loss

        return gate_scores, None


class SwitchMoE(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture.

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(self, dim: int, hidden_dim: int, output_dim: int, num_experts: int, \
                    top_k: int, dropout: int, capacity_factor: float = 1.0, mult: int = 2, \
                    use_aux_loss: bool = False, *args, **kwargs):
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.mult = mult
        # self.use_aux_loss = use_aux_loss
        self.use_aux_loss = False
        
        # self.experts = nn.ModuleList(
        #     [
        #         # FeedForward(dim, dim, mult, *args, **kwargs)
        #         nn.Sequential(
        #             nn.Conv1d(in_channels=dim, out_channels=hidden_dim, kernel_size=1),
        #             nn.LayerNorm(hidden_dim),
        #             nn.Conv1d(in_channels=hidden_dim, out_channels=dim, kernel_size=1),
        #             nn.LayerNorm(dim),
        #             nn.Dropout(dropout),
        #             nn.GELU()
        #         )
        #         for _ in range(num_experts)
        #     ]
        # )
        self.conv1 = nn.Conv1d(in_channels=dim, out_channels=hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=dim, kernel_size=1)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
        
        self.gate = SwitchGate(
            dim,
            num_experts,
            top_k,
            capacity_factor,
        )

    def forward(self, x: Tensor):
        """
        Forward pass of the SwitchMoE module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        # (batch_size, seq_len, num_experts)
        # print(x.shape)
        gate_scores, loss = self.gate(
            x, use_aux_loss=self.use_aux_loss
        )
        # print("loss: ", loss)
        # print("gate_scores", gate_scores)
        expert_outputs = []
        for i in range(self.num_experts):
            stacked_tensor = x.transpose(0, 1)
            tensors = []
            for i in range(stacked_tensor.shape[0]):
                tensors.append(self.dropout(self.activation(self.conv1(stacked_tensor[i].transpose(-1,1)))))
            stacked_tensor = torch.stack(tensors)
            # print("stacked_tensor: ", stacked_tensor.shape)
            
            tensors = []
            for i in range(stacked_tensor.shape[0]):
                tensors.append(self.dropout(self.conv2(stacked_tensor[i]).transpose(-1,1)))
            stacked_tensor = torch.stack(tensors)
            # print("stacked_tensor: ", stacked_tensor.shape)
            
            expert_output = stacked_tensor.transpose(0, 1)
            expert_outputs.append(expert_output)

            
        # Dispatch to experts
        # expert_outputs = [expert(x) for expert in self.experts]

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(
            expert_outputs, dim=-1
        )  # (batch_size, seq_len, output_dim, num_experts)
        # print(stacked_expert_outputs.shape)
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[
                torch.isnan(stacked_expert_outputs)
            ] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        # print("moe_output shape: ", moe_output.shape)
        # print("loss: ", loss)
        # print("moe_output: ", moe_output)
        return moe_output, loss

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x: B, ts_d, L, d_model
        """
        batch_size, ts_d, seg_num, d_model = x.shape
        pad_num = seg_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        x = torch.cat(seg_to_merge, -1)  # [B, ts_d, seg_num/win_size, win_size*d_model]

        x = self.norm(x)
        x = self.linear_trans(x)

        return x

class scale_block(nn.Module):
    '''
    We can use one segment merging layer followed by multiple TSA layers in each scale
    the parameter `depth' determines the number of TSA layers used in each scale
    We set depth = 1 in the paper
    '''
    def __init__(self, win_size, attn, num_experts, top_k, d_model, n_heads, d_ff, depth, dropout, \
                    seg_num = 10, factor=10):
        super(scale_block, self).__init__()

        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        
        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(seg_num, factor, attn, d_model, n_heads, \
                                                        d_ff, dropout))
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu
        
        self.ffn = SwitchMoE(
            d_model, d_ff, d_model, num_experts, top_k, dropout
        )
    
    def forward(self, x):
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)
        
        for layer in self.encode_layers:
            new_x = layer(x)
            # print("new_x: ", new_x.shape)
            x = x + self.dropout(new_x)
            # print("x: ", x.shape)
            y = x = self.norm1(x)
            
            
            # print("y and x: ", x.shape)
            # torch.Size([32, 7, 8, 256])
            # torch.Size([32, 288, 256]) batch_size length feature
            # 将列表中的张量堆叠成一个新的张量，形状为 [7, 32, 256, 8]
            y, _ = self.ffn(y)
            # stacked_tensor = y.transpose(0, 1)
            # # print("stacked_tensor", stacked_tensor[0].shape)

            # # 将形状为 [7, 32, 256, 8] 的张量传递给卷积层
            # tensors = []
            # for i in range(stacked_tensor.shape[0]):
            #     tensors.append(self.dropout(self.activation(self.conv1(stacked_tensor[i].transpose(-1,1)))))
            # stacked_tensor = torch.stack(tensors)
            # # print("stacked_tensor: ", stacked_tensor.shape)
            
            # tensors = []
            # for i in range(stacked_tensor.shape[0]):
            #     tensors.append(self.dropout(self.conv2(stacked_tensor[i]).transpose(-1,1)))
            # stacked_tensor = torch.stack(tensors)
            # # print("stacked_tensor: ", stacked_tensor.shape)
            
            # y = stacked_tensor.transpose(0, 1)
            # print("y: ", y.shape)

            # y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
            # # print("y: ", y.shape)
            # y = self.dropout(self.conv2(y).transpose(-1,1))
        
        return self.norm2(x+y)        
        
        # return x

class Encoder(nn.Module):
    '''
    The Encoder of Crossformer.
    '''
    def __init__(self, e_blocks, win_size, attn, num_experts, top_k, d_model, n_heads, d_ff, block_depth, dropout,
                in_seg_num = 10, factor=10):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList()

        self.encode_blocks.append(scale_block(1, attn, num_experts, top_k, d_model, n_heads, d_ff, block_depth, dropout,\
                                            in_seg_num, factor))
        for i in range(1, e_blocks):
            self.encode_blocks.append(scale_block(win_size, attn, num_experts, top_k, d_model, n_heads, d_ff, block_depth, dropout,\
                                            ceil(in_seg_num/win_size**i), factor))

    def forward(self, x):
        encode_x = []
        encode_x.append(x)
        
        for block in self.encode_blocks:
            x = block(x)
            encode_x.append(x)

        return encode_x