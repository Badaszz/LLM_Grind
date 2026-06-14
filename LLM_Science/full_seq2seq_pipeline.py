import pandas as pd
import torch
import torch.nn as nn
# import kagglehub
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import os

# # --- Device Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # # --- Data Preparation 
# print("Downloading dataset...")
# path = kagglehub.dataset_download("olaolabenjo/englishyoruba-parallel-corpus-sample")
# print(f"Dataset downloaded to: {path}")

df = pd.read_csv("yoruba_english_parallel_sample.csv")
english_sentences = df["english"].astype(str)
yoruba_sentences = df["yoruba"].astype(str)

# Create sample_data directory if it doesn't exist
if not os.path.exists("/content/sample_data"):
    os.makedirs("/content/sample_data")

# Save english_sentences and yoruba_sentences to temporary files for CBOW training
english_sentences.to_csv("/content/sample_data/english.txt", index=False, header=False)
yoruba_sentences.to_csv("/content/sample_data/yoruba.txt", index=False, header=False)
print("English and Yoruba sentences saved to /content/sample_data/.")


# --- Vocabulary Building Functions ---
def build_vocab(sentences: list, min_freq: int = 1):
    tokens = []
    for s in sentences:
        tokens.extend(s.lower().split())
    freq = Counter(tokens)
    vocab = {word: idx for idx, (word, c) in enumerate(freq.items()) if c >= min_freq}
    return vocab

def build_vocab_seq2seq(sentences, min_freq=1, specials=["<pad>", "<sos>", "<eos>"]):
    counter = Counter()
    for s in sentences:
        counter.update(s.split())

    vocab = {tok: idx for idx, tok in enumerate(specials)}
    idx = len(specials)

    for word, freq in counter.items():
        if freq >= min_freq and word not in vocab:
            vocab[word] = idx
            idx += 1

    return vocab

# --- CBOW Model and Training (Adapted for GPU) ---
class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embeds = self.emb(x)          # [B, context, D]
        mean = embeds.mean(dim=1)     # CBOW
        return self.linear(mean)

class CBOWDataset(Dataset):
    def __init__(self, sentences: list, vocab: dict, window: int =2):
        self.data = []
        self.vocab = vocab

        for s in sentences:
            words = s.lower().split()
            for i in range(window, len(words) - window):
                context = words[i-window:i] + words[i+1:i+window+1]
                target = words[i]

                if target in vocab and all(w in vocab for w in context):
                    self.data.append(
                        ([vocab[w] for w in context], vocab[target])
                    )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context), torch.tensor(target)

def train_cbow(sentences, save_path: str, embedding_dim: int = 50, epochs: int = 1000):
    vocab = build_vocab(sentences)
    dataset = CBOWDataset(sentences, vocab)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = CBOW(len(vocab), embed_dim=embedding_dim).to(device) # Move model to device
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"Starting CBOW training for {save_path}...")
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device) # Move tensors to device
            loss = loss_fn(model(x), y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch} Loss {loss.item():.4f}") # Comment out to reduce output

    torch.save({
        "embeddings": model.emb.weight.data.cpu(), # Save embeddings back to CPU
        "vocab": vocab
    }, save_path)
    print(f"CBOW embeddings saved to {save_path}")


# --- Seq2Seq Model Classes (num_layers=4 and GPU adaptation for Encoder output) ---
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=4): # num_layers=4
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Add two parallel bidirectional LSTMs
        self.lstm1 = nn.LSTM(emb_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(emb_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        emb = self.embedding(x)

        # Process through both LSTMs
        _, (h1, c1) = self.lstm1(emb)
        _, (h2, c2) = self.lstm2(emb)

        # Initialize combined hidden and cell states on the same device as input 'x'
        batch_size = x.size(0)
        combined_h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
        combined_c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)

        # Combine hidden and cell states
        for i in range(self.num_layers):
            # Sum forward and backward components for lstm1 for current layer
            h1_layer_combined = h1[2 * i, :, :] + h1[2 * i + 1, :, :]
            c1_layer_combined = c1[2 * i, :, :] + c1[2 * i + 1, :, :]

            # Sum forward and backward components for lstm2 for current layer
            h2_layer_combined = h2[2 * i, :, :] + h2[2 * i + 1, :, :]
            c2_layer_combined = c2[2 * i, :, :] + c2[2 * i + 1, :, :]

            # Sum the combined states from both LSTMs
            combined_h[i, :, :] = h1_layer_combined + h2_layer_combined
            combined_c[i, :, :] = c1_layer_combined + c2_layer_combined

        return combined_h, combined_c

class Decoder(nn.Module):
  def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers=4): # num_layers=4
      super().__init__()
      self.embedding = nn.Embedding(vocab_size, emb_dim)
      self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True) # num_layers=4
      self.fc = nn.Linear(hidden_dim, vocab_size)

  def forward(self, x, h, c):
      emb = self.embedding(x)
      outputs, (h, c) = self.lstm(emb, (h, c))
      return self.fc(outputs), h, c


# --- Seq2Seq Dataset and Dataloader ---
class seq2seqDataset(Dataset):
    def __init__(self, eng, yor, eng_vocab, yor_vocab):
        self.pairs = []
        for e, y in zip(eng, yor):
            e_tokens = e.lower().split()
            y_tokens = y.lower().split()

            # Filter for words in vocab and add <sos>, <eos>
            e_ids = [eng_vocab["<sos>"]] + \
                    [eng_vocab[w] for w in e_tokens if w in eng_vocab] + \
                    [eng_vocab["<eos>"]]

            y_ids = [yor_vocab["<sos>"]] + \
                    [yor_vocab[w] for w in y_tokens if w in yor_vocab] + \
                    [yor_vocab["<eos>"]]

            self.pairs.append((e_ids, y_ids))

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0], dtype=torch.long), \
               torch.tensor(self.pairs[idx][1], dtype=torch.long)

    def __len__(self):
        return len(self.pairs)

def collate_batch(batch):
    src_list, tgt_list = [], []
    for _src, _tgt in batch:
        src_list.append(_src)
        tgt_list.append(_tgt)

    # Pad sequences to the length of the longest sequence in the batch
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=eng_vocab_seq2seq["<pad>"])
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_list, batch_first=True, padding_value=yor_vocab_seq2seq["<pad>"])
    return src_padded, tgt_padded


# --- Seq2Seq Training Function (Adapted for GPU) ---
def train_seq2seq(encoder, decoder, loader, epochs: int = 1000):
    loss_fn = nn.CrossEntropyLoss(ignore_index=yor_vocab_seq2seq["<pad>"]) # Ignore padding in loss calculation
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

    print("Starting seq2seq training...")
    for epoch in range(epochs):
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device) # Move tensors to device

            h, c = encoder(src)
            output, _, _ = decoder(tgt[:, :-1], h, c)

            loss = loss_fn(
                output.reshape(-1, output.shape[-1]),
                tgt[:, 1:].reshape(-1)
            )

            optim.zero_grad()
            loss.backward()
            optim.step()

        print(f"Epoch {epoch} Loss {loss.item():.4f}")


# --- Translation Function (Adapted for GPU) ---
def translate(sentence, encoder, decoder, eng_vocab, yor_vocab, max_len=20):
    encoder.eval()
    decoder.eval()
    inv_vocab = {v:k for k,v in yor_vocab.items()}

    src = torch.tensor([[eng_vocab[w] for w in sentence.lower().split() if w in eng_vocab]], device=device) # Move input to device
    h, c = encoder(src)

    token = torch.tensor([[yor_vocab["<sos>"]]], device=device) # Move input to device
    result = []

    for _ in range(max_len):
        out, h, c = decoder(token, h, c)
        idx = out.argmax(-1).item()
        if inv_vocab[idx] == "<eos>":
            break
        if inv_vocab[idx] == "<pad>":
            token = torch.tensor([[idx]], device=device) # Ensure new token is on device
            continue
        result.append(inv_vocab[idx])
        token = torch.tensor([[idx]], device=device) # Ensure new token is on device

    encoder.train()
    decoder.train()
    return " ".join(result)


# --- Main Execution Flow ---

# Build vocabularies for seq2seq
eng_vocab_seq2seq = build_vocab_seq2seq(english_sentences)
yor_vocab_seq2seq = build_vocab_seq2seq(yoruba_sentences)
print(f"English seq2seq vocabulary size: {len(eng_vocab_seq2seq)}")
print(f"Yoruba seq2seq vocabulary size: {len(yor_vocab_seq2seq)}")


# Use CBOW to learn embeddings and save them
EMBD_DIM = 64
train_cbow(open("/content/sample_data/english.txt").readlines(), "/content/sample_data/eng_embeddings.pt", embedding_dim=EMBD_DIM, epochs = 1000)
train_cbow(open("/content/sample_data/yoruba.txt").readlines(), "/content/sample_data/yor_embeddings.pt", embedding_dim=EMBD_DIM, epochs = 1000)

# Load embeddings
eng_embed_data = torch.load("/content/sample_data/eng_embeddings.pt")
yor_embed_data = torch.load("/content/sample_data/yor_embeddings.pt")

eng_embeddings = eng_embed_data["embeddings"]
yor_embeddings = yor_embed_data["embeddings"]
print("Embeddings loaded.")

HIDDEN_DIM = 128
NUM_LAYERS = 2

# Initialize the encoder and the decoder with num_layers=4 and move to device
encoder = Encoder(
    vocab_size=len(eng_vocab_seq2seq),
    emb_dim=EMBD_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS # Explicitly set num_layers to 4
).to(device)

decoder = Decoder(
    vocab_size=len(yor_vocab_seq2seq),
    emb_dim=EMBD_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS # Explicitly set num_layers to 4
).to(device)
print(f"Encoder and Decoder initialized with {encoder.num_layers} layers and moved to {device}.")


# Load embeddings into the encoder and decoder
encoder.embedding.weight.data[:eng_embeddings.size(0)] = eng_embeddings.to(device)
decoder.embedding.weight.data[:yor_embeddings.size(0)] = yor_embeddings.to(device)

# Freeze embedding layers
encoder.embedding.weight.requires_grad = False
decoder.embedding.weight.requires_grad = False
print("Embeddings loaded into models and frozen.")


# Create the seq2seq Dataset and DataLoader
dataset = seq2seqDataset(
    english_sentences,
    yoruba_sentences,
    eng_vocab_seq2seq,
    yor_vocab_seq2seq
)

BATCH_SZE = 16

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SZE,
    shuffle=False,
    collate_fn=collate_batch
)
print("Seq2seq dataset and dataloader created.")

# Train seq2seq model with increased layers and GPU
train_seq2seq(
    encoder=encoder,
    decoder=decoder,
    loader=loader,
    epochs= 300 # Retrain for 300 epochs
)
print("Seq2seq model training complete.")

# Evaluate translated output
sample_sentence = "If the devil could organize to fight in heaven"
translation = translate(
    sample_sentence,
    encoder,
    decoder,
    eng_vocab_seq2seq,
    yor_vocab_seq2seq
)

print(f"\nOriginal English: {sample_sentence}")
print(f"Translated Yoruba: {translation}")