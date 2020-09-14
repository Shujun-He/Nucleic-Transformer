import math
import torch
import torch.nn as nn
import torch.nn.functional as F


#mish activation
class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
        return x *( torch.tanh(F.softplus(x)))


from torch.nn.parameter import Parameter
def gem(x, p=3, eps=1e-6):
    return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p)
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()


    def forward(self, src , src_mask = None, src_key_padding_mask = None):
        src2,attention_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attention_weights


class LinearDecoder(nn.Module):
    def __init__(self,num_classes,ninp,dropout,pool=True,):
        super(LinearDecoder, self).__init__()
        # if pool:
            # self.pool_layer=GeM()
        if pool:
            self.classifier=nn.Linear(ninp,num_classes)
        else:
            self.classifier=nn.Linear(ninp,num_classes)
        self.pool=pool
        self.pool_layer=GeM()

    def forward(self,x):
        if self.pool:
            # max_x,_=torch.max(x,dim=1)
            # x=torch.cat([torch.mean(x,dim=1),max_x],dim=-1)
            #print(x.shape)
            x=self.pool_layer(x.permute(0,2,1)).permute(0,2,1).squeeze()
            #print(x.shape)
        x=self.classifier(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class K_mer_aggregate(nn.Module):
    def __init__(self,kmers,in_dim,out_dim,dropout=0.1,stride=1):
        super(K_mer_aggregate, self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.convs=[]
        for i in kmers:
            print(i)
            self.convs.append(nn.Conv1d(in_dim,out_dim,i,stride=stride,padding=0))
        self.convs=nn.ModuleList(self.convs)
        self.activation=nn.ReLU(inplace=True)
        #self.activation=Mish()

    def forward(self,x):
        outputs=[]
        for conv in self.convs:
            outputs.append(self.dropout(self.activation(conv(x))))
        outputs=torch.cat(outputs,dim=2)
        return outputs


class NucleicTransformer(nn.Module):

    def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, kmer_aggregation, kmers, stride=1,dropout=0.5):
        super(NucleicTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.kmers=kmers
        #if self.ngrams!=None:
        self.kmer_aggregation=kmer_aggregation
        if self.kmer_aggregation:
            self.k_mer_aggregate=K_mer_aggregate(kmers,ninp,ninp,stride=stride)
        else:
            print("No kmer aggregation is chosen")
        self.transformer_encoder = []
        for i in range(nlayers):
            self.transformer_encoder.append(TransformerEncoderLayer(ninp, nhead, nhid, dropout))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(ntoken, ninp)
        #self.directional_encoder = nn.Embedding(3, ninp//8)
        self.ninp = ninp
        self.decoder = LinearDecoder(nclass,ninp,dropout)
        # self.recon_decoder = LinearDecoder(ntoken,ninp,dropout,pool=False)
        # self.error_decoder = LinearDecoder(2,ninp,dropout,pool=False)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        src = src.permute(1,0)
        #dir = dir.permute(1,0)
        src = self.encoder(src) #* math.sqrt(self.ninp)
        #dir = self.directional_encoder(dir)
        #src = torch.cat([src,dir],dim=-1)
        src = self.pos_encoder(src)
        #if self.ngrams!=None:
        if self.kmer_aggregation:
            src = self.k_mer_aggregate(src.permute(1,2,0)).permute(2,0,1)
            #print(src.shape)
            #src = torch.cat([src,kmer_output],dim=0)
            #src = kmer_output
        attention_weights=[]
        for layer in self.transformer_encoder:
            src,attention_weights_layer=layer(src)
            attention_weights.append(attention_weights_layer)
        #attention_weights=torch.stack(attention_weights).permute(1,0,2,3)
        encoder_output = src.permute(1,0,2)
        #print(encoder_output.shape)
        output = self.decoder(encoder_output)
        # recon_src = self.recon_decoder(encoder_output)
        # error_src = self.error_decoder(encoder_output)
        return output
