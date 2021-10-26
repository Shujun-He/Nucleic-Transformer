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


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.gamma=nn.Parameter(torch.tensor(100.0))

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            #attn = attn.masked_fill(mask == 0, -1e9)
            #attn = attn#*self.gamma
            attn = attn+mask*self.gamma
        attn = self.dropout(F.softmax(attn, dim=-1))

        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask  # For head axis broadcasting

        q, attn = self.attention(q, k, v, mask=mask)
        #print(attn.shape)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        #print(q.shape)
        #exit()
        # q = self.dropout(self.fc(q))
        # q += residual

        #q = self.layer_norm(q)

        return q, attn

class ConvTransformerEncoderLayer(nn.Module):
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

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, k = 3):
        super(ConvTransformerEncoderLayer, self).__init__()
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention(d_model, nhead, d_model//nhead, d_model//nhead, dropout=dropout)
        # self.mask_conv1 = nn.Conv2d(1,d_model,k)
        # self.mask_activation1=nn.ReLU()
        # self.mask_conv2 = nn.Conv2d(d_model,nhead,1)
        self.mask_conv1 = nn.Sequential(nn.Conv2d(nhead//4,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(nhead//4,nhead,1),
                                        # nn.BatchNorm2d(nhead),
                                        # nn.ReLU(inplace=True),
                                        )
        self.mask_deconv = nn.Sequential(nn.ConvTranspose2d(nhead,nhead//4,k),
                                        nn.BatchNorm2d(nhead//4),
                                        nn.Sigmoid()
                                        )
        # self.mask_activation2=nn.ReLU()
        # self.mask_conv3 = nn.Conv2d(d_model//2,nhead,1)
        #torch.nn.init.ones_(self.mask_conv.weight)
        #self.mask_conv.weight.requires_grad=False
        #self.mask_conv.weight.requires_grad=False
        #self.mask_conv.requires_grad=False
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        #self.dropout4 = nn.Dropout(dropout)


        self.activation = nn.ReLU()

        self.conv=nn.Conv1d(d_model,d_model,k,padding=0)
        self.deconv=nn.ConvTranspose1d(d_model,d_model,k)

    def forward(self, src , mask):
        res = src
        src = self.norm3(self.conv(src.permute(0,2,1)).permute(0,2,1))
        mask_res=mask
        mask = self.mask_conv1(mask)
        #mask = self.mask_activation1(mask)
        #mask = self.mask_conv2(mask)
        # mask = self.mask_activation2(mask)
        # mask = self.mask_conv3(mask)
        src2,attention_weights = self.self_attn(src, src, src, mask=mask)
        #src3,_ = self.self_attn(src, src, src, mask=None)
        #src2=src2+src3
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = res + self.dropout3(self.deconv(src.permute(0,2,1)).permute(0,2,1))
        src = self.norm4(src)
        mask = self.mask_deconv(mask)+mask_res
        return src,attention_weights,mask

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
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





class NucleicTransformer(nn.Module):

    def __init__(self, ntoken, nclass, ninp, nhead, nhid, nlayers, kmer_aggregation, kmers, stride=1,dropout=0.5,pretrain=False,return_aw=False):
        super(NucleicTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        #self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.kmers=kmers
        #if self.ngrams!=None:
        self.transformer_encoder = []
        for i in range(nlayers):
            self.transformer_encoder.append(ConvTransformerEncoderLayer(ninp, nhead, nhid, dropout, k=kmers[0]-i))
        self.transformer_encoder= nn.ModuleList(self.transformer_encoder)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.projection = nn.Linear(ninp*3, ninp)
        #self.directional_encoder = nn.Embedding(3, ninp//8)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp,nclass)
        self.mask_dense=nn.Conv2d(4,nhead//4,1)
        # self.recon_decoder = LinearDecoder(ntoken,ninp,dropout,pool=False)
        # self.error_decoder = LinearDecoder(2,ninp,dropout,pool=False)

        self.return_aw=return_aw
        self.pretrain=pretrain


        self.pretrain_decoders=nn.ModuleList()
        self.pretrain_decoders.append(nn.Linear(ninp,4))
        self.pretrain_decoders.append(nn.Linear(ninp,3))
        self.pretrain_decoders.append(nn.Linear(ninp,7))

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, mask=None):
        B,L,_=src.shape
        src = src
        src = self.encoder(src).reshape(B,L,-1)
        src =self.projection(src)
        #src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2)

        #mask=mask.unsqueeze(1)
        mask=self.mask_dense(mask)
        for layer in self.transformer_encoder:
            src,attention_weights_layer,mask=layer(src, mask)
            #attention_weights.append(attention_weights_layer)
        #attention_weights=torch.stack(attention_weights).permute(1,0,2,3)
        encoder_output = src
        #print(deconved.shape)
        #print(encoder_output.shape)
        output = self.decoder(encoder_output)
        # recon_src = self.recon_decoder(encoder_output)
        # error_src = self.error_decoder(encoder_output)
        if self.pretrain:
            ae_outputs=[]
            for decoder in self.pretrain_decoders:
                ae_outputs.append(decoder(encoder_output))
            return ae_outputs
        else:
            if self.return_aw:
                return output,attention_weights_layer
            else:
                return output
