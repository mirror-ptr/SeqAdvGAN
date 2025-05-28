from .IrisMapEncoder import IrisMapEncoder
import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .PositionalEncoding import PositionalEncoding

class IrisBabelTransformer(nn.Module):
    def __init__(self, input_dim=4096, action_dim=4, num_decoder_head=8, num_decoder_layer=4, max_len=1024):
        super(IrisBabelTransformer, self).__init__()
       # IrisBabelDecoder
        # 显式设置 nhead 为 8，以确保与 d_model=512 兼容
        # 错误信息中的 24 可能是一个硬编码或错误传递的值
        # 假设 IrisBabel 的设计是使用 8 个注意力头
        decoder_layer = TransformerDecoderLayer(d_model=input_dim, nhead=8, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layer)
        self.projection = nn.Linear(in_features=input_dim, out_features=action_dim)
        # 确保 self.num_decoder_head 与 TransformerDecoderLayer 的 nhead 一致
        self.num_decoder_head = 8
        self.num_decoder_layer = num_decoder_layer
        self.positional_encoding = PositionalEncoding(d_model=input_dim ,max_len=max_len+1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_seq, target_seq):
        # src_seq is the memory (K, V) from the encoder/perception layer
        # target_seq is the query (Q) for the decoder, which is the evolving output
        
        # Add positional encoding to both source and target sequences
        # Using the correct positional encoding attribute name
        src_seq_pos = self.positional_encoding(src_seq)
        target_seq_pos = self.positional_encoding(target_seq)
        
        # Standard nn.TransformerDecoder usage: decoder(tgt, memory)
        # The TransformerDecoder forward method returns a single tensor: the output sequence
        output = self.decoder(target_seq_pos, src_seq_pos)
        
        # The forward method in this custom model does not return attention weights
        # We will retrieve them later if needed. For now, we return None.
        attention_probs = None 
        
        return output, attention_probs
