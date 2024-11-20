import numpy as np
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C = 2 # C * 2 + 1
K = 2
VOCAB_SIZE = 30000
UNK = "<UNK>"
word_vec_dim = 200
batch_size = 128
epochs = 2
learning_rate = 1e-2

with open("data/text8.train.txt", "r", encoding='utf-8') as f:
    text = f.read()

# print(text[:100])

def my_tokenizer(text):
    text_list = text.split(" ")
    return text_list

text_list = my_tokenizer(text)

counter = Counter(text_list).most_common(VOCAB_SIZE)
# print(counter)

counter = dict(counter)

# idx2word: ["a", "the", "of", ...]
# word2idx: {"a": 0, "the": 1, "of": 2, ...}
idx2word = [k for k in counter.keys()]
idx2word = idx2word + [UNK]
# idx2word = list(counter.keys())
word2idx = {word:idx for idx, word in enumerate(idx2word)}

word_freq = list(counter.values())
word_freq.append(len(text_list) - np.sum(word_freq))
word_freq = np.array(word_freq, dtype=np.float64)
word_freq = word_freq ** (3.0 / 4.0)
word_freq = word_freq / np.sum(word_freq)

class SkipGramDataset(Dataset):
    def __init__(self, text_list, idx2word, word2idx, word_freq):
        text_index = [word2idx.get(word, word2idx[UNK]) for word in text_list]
        self.text_tensor = torch.tensor(text_index, dtype=torch.long)
        self.word_freq = torch.tensor(word_freq, dtype=torch.float64)

    def __len__(self):
        return self.text_tensor.shape[0]

    def __getitem__(self, index):
        center_word = self.text_tensor[index]
        center_word = torch.tensor(center_word, dtype=torch.long)
        # center_word: 5 -> [3, 4, 6, 7]
        context_index = list(range(index - C, index)) + list(range(index + 1, index + C + 1))
        pos_words = self.text_tensor[context_index]
        neg_words = torch.multinomial(self.word_freq, C * 2 * K, replacement=False)

        return center_word, pos_words, neg_words

# skip-gram model
class SkipGramModel(torch.nn.Module):
    def __init__(self, vocab_size, word_vec_dim):
        super(SkipGramModel, self).__init__()
        self.in_embedding = torch.nn.Embedding(vocab_size, word_vec_dim)
        self.out_embedding = torch.nn.Embedding(vocab_size, word_vec_dim)
        init_value = 0.5 / word_vec_dim
        self.in_embedding.weight.data.normal_(-init_value, init_value)
        self.out_embedding.weight.data.normal_(-init_value, init_value)

    def forward(self, center_word, pos_words, neg_words):
        '''
        :param center_word: [B]
        :param pos_words: [B, 2 * C]
        :param neg_words: [B, 2 * C * K]
        :return: loss
        '''
        center_word_emb = self.in_embedding(center_word) # [B, d]
        pos_words_emb = self.out_embedding(pos_words) # [B, 2 * C, d]
        neg_words_emb = self.out_embedding(neg_words) # [B, 2 * C * K, d]

        # bmm [B, 2 * C, d] x [B, d, 1] = [B, 2 * C, 1]
        pos_logits = torch.bmm(pos_words_emb, torch.unsqueeze(center_word_emb, 2))
        pos_logits = pos_logits.squeeze()

        # bmm [B, 2 * C * K, d] x [B, d, 1] = [B, 2 * C * K, 1]
        neg_logits = torch.bmm(neg_words_emb, -torch.unsqueeze(center_word_emb, dim=2))
        neg_logits = neg_logits.squeeze()

        pos_logits = torch.nn.functional.logsigmoid(pos_logits) # [B, 2 * C]
        neg_logits = torch.nn.functional.logsigmoid(neg_logits) # [B, 2 * C * K]

        pos_logits = torch.sum(pos_logits, dim=1) # [B]
        neg_logits = torch.sum(neg_logits, dim=1) # [B]

        loss = pos_logits + neg_logits # [B]

        loss = torch.mean(loss)
        loss = -loss
        return loss

# train skip-gram
dataset = SkipGramDataset(text_list, idx2word, word2idx, word_freq)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = SkipGramModel(VOCAB_SIZE + 1, word_vec_dim)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        batch_center_word, batch_pos_words, batch_neg_words = batch
        batch_center_word = batch_center_word.to(device)
        batch_pos_words = batch_pos_words.to(device)
        batch_neg_words = batch_neg_words.to(device)

        loss = model(batch_center_word, batch_pos_words, batch_neg_words)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, step, loss.item()))

print('finish')



