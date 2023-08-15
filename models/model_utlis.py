import torch
from torch import nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        # Normalize input_tensor
        s = (x - u).pow(2).mean(-1, keepdim=True)
        # Apply scaling and bias
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        # input_ids = input_ids.unsqueeze(0)
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)  
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)  
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs_0


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, drugA, drugB, drugA_attention_mask):
        # update drugA
        mixed_query_layer = self.query(drugA)
        mixed_key_layer = self.key(drugB)
        mixed_value_layer = self.value(drugB)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if drugA_attention_mask == None:
            attention_scores = attention_scores
        else:
            attention_scores = attention_scores + drugA_attention_mask

        attention_probs_0 = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs_0)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)       

        return context_layer, attention_probs_0


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states    
        

class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs_0 = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs_0    


class Attention_SSA(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention_SSA, self).__init__()
        self.self = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask):
        drugA_self_output, attention_probs_0 = self.self(drugA, drugB, drugA_attention_mask)
        drugA_attention_output = self.output(drugA_self_output, drugA)
        return drugA_attention_output, attention_probs_0 
    

class Attention_CA(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention_CA, self).__init__()
        self.self = CrossAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask, drugB_attention_mask):
        drugA_self_output, drugA_attention_probs_0 = self.self(drugA, drugB, drugA_attention_mask)
        drugB_self_output, drugB_attention_probs_0 = self.self(drugB, drugA, drugB_attention_mask)
        drugA_attention_output = self.output(drugA_self_output, drugA)
        drugB_attention_output = self.output(drugB_self_output, drugB)
        return drugA_attention_output, drugB_attention_output, drugA_attention_probs_0, drugB_attention_probs_0  


class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Drug self-attention encoder
class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs_0 = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs_0  

# Cell self-attention encoder
class EncoderCell(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderCell,self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention = Attention(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)        
        self.dense = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                    nn.Linear(intermediate_size, hidden_size))


    def forward(self, hidden_states, attention_mask):
        hidden_states_1 = self.LayerNorm(hidden_states)
        attention_output, attention_probs_0  = self.attention(hidden_states_1, attention_mask)
        hidden_states_2 = hidden_states_1 + attention_output
        hidden_states_3 = self.LayerNorm(hidden_states_2)

        hidden_states_4 = self.dense(hidden_states_3)

        layer_output = hidden_states_2 + hidden_states_4

        return layer_output, attention_probs_0   


# Drug-drug mutual-attention encoder
class EncoderCA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderCA, self).__init__()
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self,drugA, drugB, drugA_attention_mask, drugB_attention_mask):
        drugA_attention_output, drugB_attention_output, drugA_attention_probs_0, drugB_attention_probs_0  = self.attention_CA(drugA, drugB, drugA_attention_mask, drugB_attention_mask)
        drugA_intermediate_output = self.intermediate(drugA_attention_output)
        drugA_layer_output = self.output(drugA_intermediate_output, drugA_attention_output)
        drugB_intermediate_output = self.intermediate(drugB_attention_output)
        drugB_layer_output = self.output(drugB_intermediate_output, drugB_attention_output)
        return drugA_layer_output, drugB_layer_output, drugA_attention_probs_0, drugB_attention_probs_0 



# Cell-cell mutual-attention encoder
class EncoderCellCA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderCellCA,self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
             
        self.dense = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                    nn.Linear(intermediate_size, hidden_size))


    def forward(self, cellA, cellB, cellA_attention_mask=None, cellB_attention_mask=None):
        cellA_1 = self.LayerNorm(cellA)
        cellB_1 = self.LayerNorm(cellB)

        cellA_attention_output, cellB_attention_output, cellA_attention_probs_0, cellB_attention_probs_0= self.attention_CA(cellA, cellB, cellA_attention_mask, cellB_attention_mask)

        # cellA_output
        cellA_2 = cellA_1 + cellA_attention_output
        cellA_3 = self.LayerNorm(cellA_2)
        cellA_4 = self.dense(cellA_3)
        cellA_layer_output = cellA_2 + cellA_4

        # cellB_output
        cellB_2 = cellB_1 + cellB_attention_output
        cellB_3 = self.LayerNorm(cellB_2)
        cellB_4 = self.dense(cellB_3)
        cellB_layer_output = cellB_2 + cellB_4

        return cellA_layer_output, cellB_layer_output, cellA_attention_probs_0, cellB_attention_probs_0


# Drug-cell mutual-attention encoder
class EncoderD2C(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderD2C, self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_CA = Attention_CA(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)
        self.dense = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                    nn.Linear(intermediate_size, hidden_size))
    
    def forward(self, cell, drug, drug_attention_mask, cell_attention_mask=None):

        cell_1 = self.LayerNorm(cell)
        cell_attention_output, drug_attention_output, cell_attention_probs_0, drug_attention_probs_0= self.attention_CA(cell_1, drug, cell_attention_mask, drug_attention_mask)
        # cell_output
        cell_2 = cell_1 + cell_attention_output
        cell_3 = self.LayerNorm(cell_2)
        cell_4 = self.dense(cell_3)
        cell_layer_output = cell_2 + cell_4
        # drug_output
        drug_intermediate_output = self.intermediate(drug_attention_output)
        drug_layer_output = self.output(drug_intermediate_output, drug_attention_output)

        return cell_layer_output, drug_layer_output, cell_attention_probs_0, drug_attention_probs_0

# DrugA-drugB mutual-attention encoder, only output drugA embedding
class EncoderSSA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderSSA, self).__init__()
        self.attention_SSA = Attention_SSA(hidden_size, num_attention_heads,
                                   attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, drugA, drugB, drugA_attention_mask):
        drugA_attention_output, drugA_attention_probs_0 = self.attention_SSA(drugA, drugB, drugA_attention_mask)
        drugA_intermediate_output = self.intermediate(drugA_attention_output)
        drugA_layer_output = self.output(drugA_intermediate_output, drugA_attention_output)
        return drugA_layer_output, drugA_attention_probs_0


# CellA-cellB mutual-attention encoder, only output cellA embedding
class EncoderCellSSA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(EncoderCellSSA,self).__init__()
        self.LayerNorm = LayerNorm(hidden_size)
        self.attention_SSA = Attention_SSA(hidden_size, num_attention_heads,
                                        attention_probs_dropout_prob, hidden_dropout_prob)
             
        self.dense = nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.ReLU(),
                    nn.Dropout(hidden_dropout_prob),
                    nn.Linear(intermediate_size, hidden_size))


    def forward(self, cellA, cellB, cellA_attention_mask=None):
        cellA_1 = self.LayerNorm(cellA)
        cellB_1 = self.LayerNorm(cellB)

        cellA_attention_output, cellA_attention_probs_0 = self.attention_SSA(cellA, cellB, cellA_attention_mask)

        # cellA_output
        cellA_2 = cellA_1 + cellA_attention_output
        cellA_3 = self.LayerNorm(cellA_2)
        cellA_4 = self.dense(cellA_3)
        cellA_layer_output = cellA_2 + cellA_4

        return cellA_layer_output, cellA_attention_probs_0
