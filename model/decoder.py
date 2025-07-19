import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        outputs, _ = self.lstm(inputs)
        return self.linear(outputs)

    def sample(self, features, max_len=20):
        result = []
        inputs = features.unsqueeze(1)
        states = None

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            output = self.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            result.append(predicted.item())
            inputs = self.embed(predicted).unsqueeze(1)

        return result
