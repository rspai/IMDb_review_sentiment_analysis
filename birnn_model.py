import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        #TO-DO
        #1. Initialize Embedding Layer
        #2. Initialize RNN layer
        #3. Initialize a fully connected layer with Linear transformation
        #4. Initialize Dropout

        self.embedding = nn.Embedding(num_embeddings = vocab_size, 
                                      embedding_dim = embedding_dim, 
                                      padding_idx = pad_idx)
        self.rnn = nn.RNN(input_size = embedding_dim, 
                          hidden_size = hidden_dim, 
                          num_layers = n_layers, 
                          dropout = dropout, 
                          bidirectional = True)
        self.fc = nn.Linear(hidden_dim * n_layers, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        #text = [sent_len, batch_size]

        #TO-DO
        #1. Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        #2. Run the RNN along the sentences of length sent_len. #output = [sent len, batch size, hid dim * num directions]; #hidden = [num layers * num directions, batch size, hid dim]
        #3. Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout

        embedding_dropout = self.dropout(self.embedding(text))
        output, hidden = self.rnn(embedding_dropout)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)) 
        
        return self.fc(hidden)