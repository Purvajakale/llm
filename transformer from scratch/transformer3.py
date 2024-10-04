import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.data import Field, BucketIterator
import math
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Define your Transformer classes here...
# (Insert the previous Transformer classes here)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.final_layer = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.final_layer(dec_output)
        return output
    
def create_mask(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask
    return src_mask, tgt_mask

from torchtext.data import LabelField

# Define Fields for text and label
tokenizer = get_tokenizer('basic_english')
TEXT = Field(tokenize=tokenizer, lower=True, include_lengths=True)
LABEL = LabelField(dtype=torch.long) 

# Load the IMDb dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d", min_freq=5)
LABEL.build_vocab(train_data)  # Build vocab for labels

# Create iterators for the dataset
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data), 
    batch_size=64, 
    sort_within_batch=True, 
    sort_key=lambda x: len(x.text), 
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Update the padding index
pad_idx = TEXT.vocab.stoi[TEXT.pad_token]

print(f'Training data  size: {len(train_data)}')

# Example Transformer parameters
src_vocab_size = len(TEXT.vocab)
tgt_vocab_size = len(LABEL.vocab)  # Use the size of the label vocab
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 6
dropout = 0.1

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)

# Define the training loop
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Define the training loop
def train_step(src, tgt):
    model.train()
    optimizer.zero_grad()
    
    # Reshape tgt to have the expected shape (batch_size, 1) for the decoder
    tgt = tgt.unsqueeze(1)  # Add an additional dimension

    # Create masks
    src_mask, tgt_mask = create_mask(src, tgt[:, :-1], pad_idx=pad_idx)
    
    # Forward pass
    output = model(src, tgt[:, :-1], src_mask, tgt_mask)
    
    # Calculate loss
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
num_epochs = 5  # Set number of epochs
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_iterator:
        src, src_lengths = batch.text
        
        # Convert labels to a tensor and add a dimension
        tgt = batch.label.unsqueeze(1)  # Make sure tgt is of shape (batch_size, 1)
        
        loss = train_step(src, tgt)
        total_loss += loss
    
    avg_loss = total_loss / len(train_iterator)
    print(f"Epoch: {epoch}, Average Loss: {avg_loss:.4f}")



# for batch in train_iterator:
#     print(batch.label)
#     break 

# Training loop
# num_epochs = 5  # Set number of epochs
# for epoch in range(num_epochs):
#     for batch in train_iterator:
#         src, src_lengths = batch.text
#         # Convert labels to a tensor
#         tgt = torch.tensor([1 if label == 'pos' else 0 for label in batch.label]).unsqueeze(1)
#         loss = train_step(src, tgt)
#         print(f"Epoch: {epoch}, Loss: {loss}")


# Training loop



# Inference example (Using a sample sentence)
# def infer(model, sentence):
#     model.eval()
#     tokens = tokenizer(sentence)
#     tokens = [TEXT.vocab.stoi[token] for token in tokens]
#     src = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
#     src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)  # Create mask
#     tgt = torch.tensor([1])  # Start symbol for decoder
#     tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    
#     with torch.no_grad():
#         output = model(src, tgt.unsqueeze(0), src_mask, tgt_mask)
#         return output

# # Example inference
# example_sentence = "This movie was amazing!"
# predicted_output = infer(model, example_sentence)
# print(predicted_output)