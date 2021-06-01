# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
### embeddings ###

### embeddings ###

class positional_embedding(nn.Module):
    def __init__(self,args):
        super().__init__()
        pos = torch.arange(0,args.max_len,device = args.device).unsqueeze(1) # max_len, 1
        div = (10000**(torch.arange(0,args.d_model,device = args.device)/args.d_model)).unsqueeze(0) # 1, d_model
        self.pe = torch.zeros_like(pos/div)
        self.pe[:,0::2] = torch.sin(pos/div[:,0::2])
        self.pe[:,1::2] = torch.cos(pos/div[:,0::2])
        self.pe = self.pe.to(args.device) 
        self.dropout = nn.Dropout(args.dropout)
        
    def forward(self, input):
        # input : (bs, seq_len, d_model) -> (bs, seq_len, d_model)
        seq_len = input.size(1)
        output = input + self.pe[:seq_len,:].unsqueeze(0)  
       # print(input)
       # print(sel)
        return self.dropout(output) # (max_len, d_model)        
 
class token_embedding(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.token_embedding = nn.Embedding(args.n_vocab, args.d_model, padding_idx = args.padding_idx)
    def forward(self,input):
        # input : (bs, seq_len) -> (bs, seq_len, d_model)
        output = self.token_embedding(input) 
        return output

# model에선 ReLU모델을 활용했지만 , GeLU모델도 구현함. 
class gelu(nn.Module):
    def __init__(self):
        super().__init__()
    #gelu(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.0044715x**3))
    def forward(self,x):
        return 0.5*x*(1+torch.tanh(math.sqrt(2/math.pi)*(x+0.0044715*(x**3))))
# Multihead attention
# 1. Encoder 부 - self attention
# 2. Decoder 부 - masked self attention
# 3. Encoder - Decoder attention
# Mask만 subsequent mask + padding mask하면 됨

class multi_head_attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.d_k = self.args.d_model // self.args.n_head
        self.linear_Q = nn.Linear(args.d_model,args.d_model)
        self.linear_K = nn.Linear(args.d_model,args.d_model)
        self.linear_V = nn.Linear(args.d_model,args.d_model)
        
    def forward(self, query, key, value, mask = None):
        # input (bs, seq_len, d_model) -> (bs,seq_len,h,d_k)
        # 여기서 mask - (bs, seq_len(q), seq_len(k))
        Q = self.linear_Q(query)
        Q = Q.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous() # bs,h,seq_len,d_k
        K = self.linear_K(key) 
        K = K.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        V = self.linear_V(value) 
        V = V.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        
        next = torch.matmul(Q,K.transpose(2,3).contiguous())/math.sqrt(self.d_k)
        # padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).expand(next.size()) # bs, h, seq_len(q), seq_len(k)
            next = next.masked_fill(mask,-1e8)
        softmax = nn.Softmax(3).forward(next)
        output = torch.matmul(softmax,V) # bs, h, seq_len, d_k
        output = output.transpose(1,2).contiguous()
        output = output.reshape(-1,self.args.seq_len,self.args.d_model) # bs, seq_len, d_model
        return output
    
class multi_head_self_attention(multi_head_attention):
    def __init__(self,args):
        super().__init__(args)
    def forward(self, input, mask=None):
        return super().forward(input,input,input,mask)

class feed_forward_network(nn.Module):
    def __init__(self,args):
        super().__init__()
        
        self.f1 = nn.Linear(args.d_model,args.d_ff)
        self.f2 = nn.Linear(args.d_ff,args.d_model)
        self.dropout = nn.Dropout(args.dropout)
        self.gelu = gelu()
    def forward(self,input):
        output = self.f1(input)
        output = self.dropout(self.gelu(output))
        output = self.f2(output)
        return output

# 각 layer에서 sample의 평균과 std를 구함(feature 무관)
# 그를 이용해서 각 sample를 정규화 시킴
# scaling and shifting - 이 것이 parameter임
class layer_norm(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones((1,args.seq_len,args.d_model))) # 1로 두는 까닭은 batch 마다 다를 필요가 없다.
        self.beta = nn.Parameter(torch.zeros((1,args.seq_len,args.d_model)))
        self.eps = 1e-8
    def forward(self,input):
        # input shape : (bs,seq_len,d_model)
        mean = input.mean(-1,keepdim=True) # bs, seq_len,1
        std = input.std(-1,keepdim=True) # bs, seq_len,1
        output = (input-mean)/(std+self.eps) # bs, seq_len, d_model
        #try:
        output = self.gamma*output+self.beta
        #except:
            #print(self.gamma.shape)
            #print(output.shape)
            #print(self.beta.shape)
        return output
    
class layer_connection(nn.Module):
    # input + dropout(layernorm(sublayer(input)))
    def __init__(self,args):
        super().__init__()
        self.layer_norm = layer_norm(args)
        self.dropout=nn.Dropout(args.dropout)
    def forward(self, sublayer, input):
        # input (bs, seq_len, d_model)
        # layer norm + dropout + residual net
        # attention is all you need 에선 , LayerNormalization(sublayer(input)+input)
        #print(sublayer(input).shape)
        output = input + self.dropout(self.layer_norm(sublayer(input)))
        return output
 
class Transformer_Encoder_Layer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layer_connection1 = layer_connection(args)
        self.mha = multi_head_self_attention(args)
        self.layer_connection2 = layer_connection(args)
        self.ffn = feed_forward_network(args)
    def forward(self, input, mask = None):
        # multi head attention
        # feed forward network
        output1 = self.layer_connection1(lambda x : self.mha(x,mask = mask), input) # 기가 막힌 technique
        output2 = self.layer_connection2(self.ffn, output1)
        return output2
    
class Transformer_Encoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.token_embedding = token_embedding(args)
        self.positional_embedding = positional_embedding(args)
        self.encoder = nn.ModuleList([Transformer_Encoder_Layer(args) for _ in range(args.n_layers)])
    def forward(self, input):
        # padding mask
        mask = input.eq(args.padding_idx).unsqueeze(1).expand(input.size(0),input.size(1),input.size(1)) # bs, seq_len1, seq_len1
        output = self.token_embedding(input)
        output = self.positional_embedding(output) # bs, seqlen, d_model
        for i in range(self.args.n_layers):
            output = self.encoder[i](output,mask)
        return output
    
class Transformer_Decoder_Layer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layer_connection1 = layer_connection(args)
        # decoder masked multi head self attention
        self.decoder_mha = multi_head_self_attention(args)
        self.layer_connection2 = layer_connection(args)
        # encoder - decoder multi head attention
        self.encoder_decoder_attn = multi_head_attention(args)
        # feed forward network
        self.ffn = feed_forward_network(args)

        self.layer_connection3 = layer_connection(args)
    def forward(self, decoder_input, encoder_output, decoder_mask = None, encoder_mask = None):
        output1 = self.layer_connection1(lambda x : self.decoder_mha(x,mask = decoder_mask), decoder_input)
        output2 = self.layer_connection2(lambda x : self.encoder_decoder_attn(query = x, key = encoder_output, value = encoder_output, mask = encoder_mask), decoder_input)
        output3 = self.layer_connection2(self.ffn, output2)
        return output3

class Transformer_Decoder(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.token_embedding = token_embedding(self.args) 
        self.positional_embedding = positional_embedding(self.args)
        self.decoder = nn.ModuleList([Transformer_Decoder_Layer(args) for _ in range(self.args.n_layers)])
    def forward(self, decoder_input, encoder_output, encoder_mask = None):
        # decoder input (bs, seq_len2)
        # encoder output (bs, seq_len1, d_model)
        # encoder mask (bs, seq_len1)
        # padding mask + subsquent mask
        padding_mask = decoder_input.eq(self.args.padding_idx).unsqueeze(1).expand(decoder_input.size(0),decoder_input.size(1),decoder_input.size(1)) # bs, seq_len2, seq_len2
        subsequent_mask = torch.triu(torch.ones(decoder_input.size(1),decoder_input.size(1),device = decoder_input.device),1).unsqueeze(0).bool() # 1,seq_len2,seq_len2
        mask = padding_mask + subsequent_mask
        
        output = self.token_embedding(decoder_input)
        output = self.positional_embedding(output) # bs, seqlen, d_model
        for i in range(self.args.n_layers):
            output = self.decoder[i](output,encoder_output,decoder_mask = mask, encoder_mask = encoder_mask)
        return output
    
class Transformer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.encoder = Transformer_Encoder(args)
        self.decoder = Transformer_Decoder(args)
        self.linear = nn.Linear(args.d_model, args.n_vocab)
    def forward(self,encoder_input,decoder_input):
        encoder_output = self.encoder.forward(encoder_input)
        encoder_mask = encoder_input.eq(self.args.padding_idx).unsqueeze(1).expand(encoder_input.size(0),encoder_input.size(1),encoder_input.size(1)) # bs, seq_len1, seq_len1
        decoder_output = self.decoder.forward(decoder_input,encoder_output,encoder_mask)
        output = self.linear.forward(decoder_output)
        return output

