import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention proposed in "Attention Is All You Need"
    Compute the dot products of the query with all keys, divide each by sqrt(dim),
    and apply a softmax function to obtain the weights on the values
    Args: dim, mask
        dim (int): dimention of attention
        mask (torch.Tensor): tensor containing indices to be masked
    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **value** (batch, v_len, d_model): tensor containing features of the encoded input sequence.
        - **mask** (-): tensor containing indices to be masked
    Returns: context, attn
        - **context**: tensor containing the context vector from attention mechanism.
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim, mid=64):
        super(ScaledDotProductAttention, self).__init__()
        self.q_branch = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=mid, kernel_size=(3,1), stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.ReLU(), 
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3,1), stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.ReLU()
        )

        self.k_branch = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=mid, kernel_size=(3,1), stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.ReLU(),    
            nn.Conv2d(in_channels=mid, out_channels=mid, kernel_size=(3,1), stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.ReLU()
        )

        self.v_branch = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=mid, kernel_size=(3,1), stride=1, padding='same', dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None),
            nn.ReLU()
        )

        self.bottleneck = nn.Conv2d(in_channels=mid, out_channels=1, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

        self.l_norm = nn.LayerNorm([mid, 1024, 1])

        self.act = nn.ReLU()

    def forward(self, x):
        query = self.q_branch(x)
        key = self.k_branch(x)
        value = self.v_branch(x)

        score = query*key

        attn = F.softmax(score, 1)
        attn = self.l_norm(attn)
        context = attn*value
        context = self.act(self.bottleneck(context))

        return context