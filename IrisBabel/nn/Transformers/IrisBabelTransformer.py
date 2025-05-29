from .IrisMapEncoder import IrisMapEncoder
import torch
from torch import nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from .PositionalEncoding import PositionalEncoding

class IrisBabelTransformer(nn.Module):
    def __init__(self, input_dim=4096, action_dim=4, num_decoder_head=8, num_decoder_layer=4, max_len=1024):
        super(IrisBabelTransformer, self).__init__()
       # IrisBabelDecoder
        decoder_layer = TransformerDecoderLayer(d_model=input_dim, nhead=num_decoder_head, batch_first=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layer)
        self.projection = nn.Linear(in_features=input_dim, out_features=action_dim)
        self.num_decoder_head = num_decoder_head
        self.num_decoder_layer = num_decoder_layer
        self.positional_encoding = PositionalEncoding(d_model=input_dim ,max_len=max_len+1)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src_seq, target_seq):
        # Generate Masks
        src_mask_unit = self.generate_square_subsequent_mask(src_seq.size(1)).to(src_seq.device)
        src_mask = torch.stack([src_mask_unit] * self.num_decoder_head * src_seq.size(0), dim=0).to(src_seq.device)
        target_mask_unit = self.generate_square_subsequent_mask(target_seq.size(1)).to(target_seq.device)
        target_mask = torch.stack([target_mask_unit] * self.num_decoder_head * target_seq.size(0), dim=0).to(target_seq.device)

        # Positional Encoding
        src_seq = self.positional_encoding(src_seq)
        target_seq = self.positional_encoding(target_seq)

        # Decoder
        output = self.decoder(target_seq, src_seq, memory_mask=src_mask, tgt_mask=target_mask, tgt_is_causal=True, memory_is_causal=True )
        return output
