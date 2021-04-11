# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
### embeddings ###

class positional_embedding(nn.Module):
    def __init__(self,args):
        super().__init__()
        pos = torch.arange(0,args.max_len,device = args.device).unsqueeze(1) # max_len, 1
        div = (10000**(torch.arange(0,args.d_model,device = args.device)/args.d_model)).unsqueeze(0) # 1, d_model
        self.pe = pos/div
        self.pe[:,0::2] = torch.sin(self.pe[:,0::2])
        self.pe[:,1::2] = torch.cos(self.pe[:,1::2])
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
class multi_head_self_attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.d_k = self.args.d_model // self.args.n_head
        self.linear_Q = nn.Linear(args.d_model,args.d_model)
        self.linear_K = nn.Linear(args.d_model,args.d_model)
        self.linear_V = nn.Linear(args.d_model,args.d_model)
        
    def forward(self, input, mask = None):
        # input (bs, seq_len, d_model) -> (bs,seq_len,h,d_k)
        # 여기서 mask는 padding mask 용 - (bs, seq_len) -> (bs,seq_len,seq_len)
        Q = self.linear_Q(input)
        Q = Q.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous() # bs,h,seq_len,d_k
        K = self.linear_K(input) 
        K = K.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        V = self.linear_V(input) 
        V = V.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        
        next = torch.matmul(Q,K.transpose(2,3).contiguous())/math.sqrt(self.d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1).expand(next.size())
            next = next.masked_fill(mask,-1e-8)
        softmax = nn.Softmax(3).forward(next)
        output = torch.matmul(softmax,V) # bs, h, seq_len, d_k
        output = output.transpose(1,2).contiguous()
        output = output.reshape(-1,self.args.seq_len,self.args.d_model) # bs, seq_len, d_model
        return output
 
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

# 2. Decoder 부 - masked self attention
class masked_multi_head_self_attention(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.d_k = self.args.d_model // self.args.n_head
        self.linear_Q = nn.Linear(args.d_model,args.d_model)
        self.linear_K = nn.Linear(args.d_model,args.d_model)
        self.linear_V = nn.Linear(args.d_model,args.d_model)
        
    def forward(self, input, mask = None):
        # input (bs, seq_len, d_model) -> (bs,seq_len,h,d_k)
        # 여기서 mask는 padding mask 용 - (bs, seq_len) -> (bs,seq_len,seq_len)
        Q = self.linear_Q(input)
        Q = Q.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous() # bs,h,seq_len,d_k
        K = self.linear_K(input) 
        K = K.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        V = self.linear_V(input) 
        V = V.reshape(-1,self.args.seq_len,self.args.n_head,self.d_k).transpose(1,2).contiguous()
        
        next = torch.matmul(Q,K.transpose(2,3).contiguous())/math.sqrt(self.d_k)
        # padding mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1).expand(next.size())
            next = next.masked_fill(mask,-1e-8)
        # subsequent mask
        subsequent_mask = torch.triu(torch.ones(self.args.seq_len,self.args.seq_len),1)
        next = next.masked_fill(subsequent_mask,-1e-8)
        softmax = nn.Softmax(3).forward(next)
        output = torch.matmul(softmax,V) # bs, h, seq_len, d_k
        output = output.transpose(1,2).contiguous()
        output = output.reshape(-1,self.args.seq_len,self.args.d_model) # bs, seq_len, d_model
        return output
 
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
    def forward(self,sublayer,input,mask = None):
        # input (bs, seq_len, d_model)
        # layer norm + dropout + residual net
        # attention is all you need 에선 , LayerNormalization(sublayer(input)+input)
        #print(sublayer(input).shape)
        if mask is None:
            output = input + self.dropout(self.layer_norm(sublayer(input)))
        else:
            output = input + self.dropout(self.layer_norm(sublayer(input,mask)))
        return output
 
class Transformer_Encoder_Layer(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.layer_connection1 = layer_connection(args)
        self.mha = multi_head_self_attention(args)
        self.layer_connection2 = layer_connection(args)
        self.ffn = feed_forward_network(args)
    def forward(self,input,mask = None):
        # multi head attention
        # feed forward network
        output1 = self.layer_connection1(self.mha, input, mask)
        output2 = self.layer_connection2(self.ffn, output1)
        return output2
 
class BERT(nn.Module):
    def __init__(self,args):
        
        super().__init__()
        self.args = args
        self.TE = token_embedding(args)
        self.SE = segment_embedding(args)
        self.PE = positional_enbedding(args)
        self.encoder = nn.ModuleList([Transformer_Encoder_Layer(args) for _ in range(args.n_layers)])
        
    def forward(self, input_ids, segment_ids):
        # input ids (bs, seq_len) <- tokens
        # segment_ids (bs,seq_len) <- segments
        # masks (bs, seq_len) <- mask된 부분
        
        
        # embedding
        te = self.TE(input_ids)
        se = self.SE(segment_ids)
        e = te+se
        output = self.PE(e)
        # masks
        masks = input_ids.eq(self.args.padding_idx)
        # output
        for i in range(self.args.n_layers):
            output = self.encoder[i](output,masks)
        return output # (bs,seq_len,d_model)
    
class MLM(nn.Module): # MASK 위치에 해당하는 벡터가 들어오면 예측
    def __init__(self,args):
        super().__init__()
        
        self.linear = nn.Linear(args.d_model,args.n_vocab)
    def forward(self,input):
        # input : (bs, seq_len, d_model)
        output = self.linear(input)
        return output
        
class BERT_NSP(nn.Module):
    # CLS로 판단
    def __init__(self,args):
        super().__init__()
    
        self.linear = nn.Linear(args.d_model,2)
    def forward(self,input):
        # input : (bs, seq_len, d_model)
        output = self.linear(input[:,0]) # CLS token
        return output
 
class BERT_pretrain(nn.Module):
    def __init__(self,args):
        super().__init__()
        #self.tokenizer = BertTokenizer.from_pretrained(args.tokenizer_file,strip_accents=False,lowercase=False)
        self.bert = BERT(args)
        self.mlm = MLM(args)
        self.nsp = BERT_NSP(args)
    def forward(self,input_ids,segment_ids):
        output = self.bert(input_ids,segment_ids)
        mlm_output=self.mlm(output)
        nsp_output=self.nsp(output)
        return mlm_output,nsp_output
