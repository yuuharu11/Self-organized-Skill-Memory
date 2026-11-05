"""Small token-attention retrieval models for the key-mechanism audit."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class M1AttentionKey(nn.Module):
    def __init__(self, slots=4, dim=96, *, token_count=512, heads=4):
        super().__init__()
        self.slots=int(slots);self.dim=int(dim);self.token_count=int(token_count)
        self.token_proj=nn.Linear(2048,dim,bias=False)
        self.position=nn.Parameter(torch.randn(token_count,dim)*.01)
        self.context=nn.Sequential(nn.Linear(79,dim),nn.GELU(),nn.Linear(dim,dim))
        self.slot=nn.Parameter(torch.randn(slots,dim)*.02)
        self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.out=nn.Sequential(nn.Linear(slots*dim,dim),nn.GELU(),nn.Linear(dim,dim))

    def encode_tokens(self,tokens,context,mask):
        # tokens [B,N,2048], context [B,79], mask True for valid patches
        h=self.token_proj(tokens)+self.position[:tokens.shape[1]][None]
        q=self.slot[None]+self.context(context)[:,None]
        pooled,weights=self.attn(q,h,h,key_padding_mask=~mask,need_weights=True,average_attn_weights=False)
        att=weights.mean(1).masked_fill(~mask[:,None],0.)
        key=F.normalize(self.out(pooled.flatten(1)),dim=-1)
        return key,h,att

    def forward(self,tokens,context,mask):
        return self.encode_tokens(tokens,context,mask)[0]


class M2SalientTokens(nn.Module):
    def __init__(self,dim=96,*,token_count=512):
        super().__init__();self.dim=int(dim);self.token_count=int(token_count)
        self.token_proj=nn.Linear(2048,dim,bias=False)
        self.position=nn.Parameter(torch.randn(token_count,dim)*.01)
        self.context=nn.Sequential(nn.Linear(79,dim),nn.GELU(),nn.Linear(dim,dim))
        self.salience=nn.Sequential(nn.Linear(dim,dim//2),nn.Tanh(),nn.Linear(dim//2,1))
        self.out=nn.Linear(dim,dim)

    def encode_tokens(self,tokens,context,mask):
        h=F.normalize(self.token_proj(tokens)+self.position[:tokens.shape[1]][None],dim=-1)
        ctx=self.context(context)[:,None]
        logits=self.salience(torch.tanh(h+ctx)).squeeze(-1).masked_fill(~mask,-1e4)
        weights=torch.softmax(logits,dim=-1).masked_fill(~mask,0.)
        key=F.normalize(self.out(torch.sum(h*weights[:,:,None],dim=1)),dim=-1)
        return key,h,weights

    def forward(self,tokens,context,mask):
        return self.encode_tokens(tokens,context,mask)[0]


class M3CandidateJudge(nn.Module):
    def __init__(self,dim=96,heads=4):
        super().__init__();self.dim=int(dim)
        self.candidate=nn.Sequential(nn.Linear(78,dim),nn.GELU(),nn.Linear(dim,dim))
        self.cross=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.score=nn.Sequential(nn.Linear(dim*2,dim),nn.GELU(),nn.Linear(dim,1))

    def forward(self,query_tokens,query_mask,candidate_actions,candidate_state):
        # q tokens [B,N,D], actions [B,C,10,7], state [B,C,8]
        b,c=candidate_actions.shape[:2]
        action=candidate_actions.reshape(b,c,70)
        cctx=self.candidate(torch.cat([action,candidate_state],dim=-1)).reshape(b*c,1,self.dim)
        q=query_tokens[:,None].expand(-1,c,-1,-1).reshape(b*c,query_tokens.shape[1],self.dim)
        mask=query_mask[:,None].expand(-1,c,-1).reshape(b*c,query_mask.shape[1])
        attended,_=self.cross(cctx,q,q,key_padding_mask=~mask,need_weights=False)
        logits=self.score(torch.cat([cctx[:,0],attended[:,0]],dim=-1)).reshape(b,c)
        return logits


class DirectActionHead(nn.Module):
    def __init__(self,dim=96):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim+79,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,70))
    def forward(self,key,context):return self.net(torch.cat([key,context],dim=-1)).reshape(-1,10,7)
