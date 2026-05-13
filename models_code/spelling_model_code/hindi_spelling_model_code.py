# =========================
# Hindi Spelling Correction Seq2Seq (Improved)
# =========================

import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from google.colab import drive
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt

# -------------------------
# 0. Configuration
# -------------------------
class Config:
    # Paths
    DRIVE_PATH = "/content/drive/MyDrive/Project DD using ML /dataset/spelling dataset/Copy of hindi_noisy_pairs.jsonl"
    SAVE_DIR = "/content/drive/MyDrive/hindi_model"

    # Data
    TRAIN_SIZE = 150_000
    VAL_SIZE = 10_000
    TEST_SIZE = 5_000
    MAX_LEN = 64  # Reduced from 128
    MIN_FREQ = 2  # Minimum character frequency

    # Model
    EMB_DIM = 256
    HID_DIM = 512
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 128
    N_EPOCHS = 10
    LR = 1e-3
    GRAD_CLIP = 1.0
    PATIENCE = 3

    # Teacher forcing schedule
    TF_START = 1.0
    TF_END = 0.5

    # Misc
    SEED = 42
    PRINT_EVERY = 100

config = Config()

# -------------------------
# 1. Setup
# -------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(config.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Mount Drive
drive.mount('/content/drive', force_remount=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)

if not os.path.exists(config.DRIVE_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {config.DRIVE_PATH}")
print(f"✅ Found dataset at: {config.DRIVE_PATH}")

# -------------------------
# 2. Load and Split Data (NO OVERLAP!)
# -------------------------
print("\n📊 Loading dataset...")
dataset = load_dataset("json", data_files=config.DRIVE_PATH, split="train", streaming=True)

# Proper split: train, val, test with NO overlap
dataset_shuffled = dataset.shuffle(seed=config.SEED, buffer_size=10000)

print("Loading validation set...")
val_data = list(dataset_shuffled.take(config.VAL_SIZE))

print("Loading test set...")
test_iter = dataset_shuffled.skip(config.VAL_SIZE)
test_data = list(test_iter.take(config.TEST_SIZE))

print("Loading training set...")
train_iter = dataset_shuffled.skip(config.VAL_SIZE + config.TEST_SIZE)
train_data = list(train_iter.take(config.TRAIN_SIZE))

print(f"✅ Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# -------------------------
# 3. Build Vocabulary (from training data only)
# -------------------------
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

print("\n📖 Building vocabulary from training data...")
char_counter = Counter()
for item in train_data:
    for key in ["noisy", "clean"]:
        if key in item and item[key]:
            char_counter.update(list(item[key]))

# Filter by frequency
frequent_chars = [ch for ch, cnt in char_counter.items() if cnt >= config.MIN_FREQ]
char_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(frequent_chars)

char2idx = {ch: i for i, ch in enumerate(char_list)}
idx2char = {i: ch for ch, i in char2idx.items()}
VOCAB_SIZE = len(char2idx)

print(f"✅ Vocabulary size: {VOCAB_SIZE}")
print(f"   Top 10 chars: {list(char_counter.most_common(10))}")

# Save vocab
vocab_save_path = os.path.join(config.SAVE_DIR, "vocab.json")
with open(vocab_save_path, "w", encoding="utf-8") as f:
    json.dump({"char2idx": char2idx, "idx2char": idx2char}, f, ensure_ascii=False, indent=2)
print(f"💾 Vocabulary saved to: {vocab_save_path}")

# -------------------------
# 4. Dataset with Dynamic Padding
# -------------------------
class SpellingDataset(Dataset):
    def __init__(self, items, char2idx, max_len=config.MAX_LEN):
        self.items = items
        self.char2idx = char2idx
        self.max_len = max_len
        self.pad_idx = char2idx[PAD_TOKEN]
        self.sos_idx = char2idx[SOS_TOKEN]
        self.eos_idx = char2idx[EOS_TOKEN]
        self.unk_idx = char2idx[UNK_TOKEN]

    def encode(self, text):
        chars = [SOS_TOKEN] + list(text)[: self.max_len - 2] + [EOS_TOKEN]
        return [self.char2idx.get(c, self.unk_idx) for c in chars]

    def __getitem__(self, idx):
        item = self.items[idx]
        src = self.encode(item.get("noisy", ""))
        trg = self.encode(item.get("clean", ""))
        return torch.tensor(src, dtype=torch.long), torch.tensor(trg, dtype=torch.long)

    def __len__(self):
        return len(self.items)

def collate_fn(batch):
    """Dynamic padding - only pad to max length in batch"""
    srcs, trgs = zip(*batch)
    srcs_padded = pad_sequence(srcs, batch_first=True, padding_value=char2idx[PAD_TOKEN])
    trgs_padded = pad_sequence(trgs, batch_first=True, padding_value=char2idx[PAD_TOKEN])
    return srcs_padded, trgs_padded

# Create datasets
train_dataset = SpellingDataset(train_data, char2idx)
val_dataset = SpellingDataset(val_data, char2idx)
test_dataset = SpellingDataset(test_data, char2idx)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                         shuffle=True, collate_fn=collate_fn, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                       shuffle=False, collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn, num_workers=2)

# -------------------------
# 5. Model Architecture (with improvements)
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=char2idx[PAD_TOKEN])
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # Combine bidirectional hidden states
        hidden = torch.tanh(self.fc_h(torch.cat((hidden[-2], hidden[-1]), dim=1))).unsqueeze(0)
        cell = torch.tanh(self.fc_c(torch.cat((cell[-2], cell[-1]), dim=1))).unsqueeze(0)

        return outputs, (hidden, cell)

class Attention(nn.Module):
    def __init__(self, enc_hid, dec_hid):
        super().__init__()
        self.attn = nn.Linear(enc_hid + dec_hid, dec_hid)
        self.v = nn.Linear(dec_hid, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        hidden = hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, enc_hid, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=char2idx[PAD_TOKEN])
        self.rnn = nn.LSTM(emb_dim + enc_hid, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim + enc_hid + emb_dim, vocab_size)
        self.attention = Attention(enc_hid, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask=None):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        attn_weights = self.attention(hidden, encoder_outputs, mask).unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        prediction = self.fc(torch.cat((output.squeeze(1),
                                       context.squeeze(1),
                                       embedded.squeeze(1)), dim=1))

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src):
        return (src != char2idx[PAD_TOKEN])

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        max_len = trg.shape[1] if trg is not None else config.MAX_LEN

        outputs = torch.zeros(batch_size, max_len, self.decoder.vocab_size).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(src)
        mask = self.create_mask(src)

        input_token = torch.full((batch_size,), char2idx[SOS_TOKEN],
                                dtype=torch.long).to(self.device)

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell,
                                               encoder_outputs, mask)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            if trg is not None and teacher_force:
                input_token = trg[:, t]
            else:
                input_token = top1

        return outputs

# -------------------------
# 6. Initialize Model
# -------------------------
print("\n🏗️  Building model...")
encoder = Encoder(VOCAB_SIZE, config.EMB_DIM, config.HID_DIM, config.DROPOUT)
decoder = Decoder(VOCAB_SIZE, config.EMB_DIM, config.HID_DIM,
                 config.HID_DIM * 2, config.DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model has {count_parameters(model):,} trainable parameters")

# -------------------------
# 7. Training Setup
# -------------------------
optimizer = optim.Adam(model.parameters(), lr=config.LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=2)
criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD_TOKEN])

# -------------------------
# 8. Metrics
# -------------------------
def compute_char_accuracy(preds, targets):
    """Character-level accuracy"""
    preds = preds.argmax(2)
    mask = targets != char2idx[PAD_TOKEN]
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0

def seq_to_text(seq, idx2char):
    """Convert sequence to text"""
    chars = []
    for idx in seq:
        idx = idx.item() if torch.is_tensor(idx) else idx
        if idx == char2idx[EOS_TOKEN]:
            break
        if idx in [char2idx[SOS_TOKEN], char2idx[PAD_TOKEN]]:
            continue
        chars.append(idx2char.get(idx, UNK_TOKEN))
    return "".join(chars)

def compute_word_accuracy(model, loader, idx2char, device):
    """Word-level accuracy (exact match)"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, teacher_forcing_ratio=0.0)

            preds = output.argmax(2).cpu().numpy()
            targets = trg.cpu().numpy()

            for pred_seq, true_seq in zip(preds, targets):
                pred_text = seq_to_text(pred_seq, idx2char)
                true_text = seq_to_text(true_seq, idx2char)
                if pred_text == true_text:
                    correct += 1
                total += 1

    return correct / total if total > 0 else 0

# -------------------------
# 9. Training & Evaluation Functions
# -------------------------
def train_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    progress_bar = tqdm(loader, desc="Training")
    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += compute_char_accuracy(output.view(-1, 1, output_dim), trg.view(-1, 1))

        progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for src, trg in tqdm(loader, desc="Evaluating"):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
            epoch_acc += compute_char_accuracy(output.view(-1, 1, output_dim), trg.view(-1, 1))

    return epoch_loss / len(loader), epoch_acc / len(loader)

# -------------------------
# 10. Training Loop with Early Stopping
# -------------------------
print("\n🎯 Starting training...\n")

best_val_loss = float('inf')
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'word_acc': []}

for epoch in range(config.N_EPOCHS):
    # Calculate teacher forcing ratio (linearly decay)
    tf_ratio = config.TF_START - (config.TF_START - config.TF_END) * (epoch / config.N_EPOCHS)

    start_time = time.time()

    train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                       criterion, config.GRAD_CLIP, tf_ratio)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    word_acc = compute_word_accuracy(model, val_loader, idx2char, device)

    end_time = time.time()

    # Update learning rate
    scheduler.step(val_loss)

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['word_acc'].append(word_acc)

    print(f"\nEpoch {epoch+1}/{config.N_EPOCHS} | Time: {end_time-start_time:.0f}s | TF: {tf_ratio:.2f}")
    print(f"  Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
    print(f"  Val Loss:   {val_loss:.3f} | Val Acc:   {val_acc:.3f}")
    print(f"  Word Acc:   {word_acc:.3f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'word_acc': word_acc,
            'config': vars(config)
        }

        model_path = os.path.join(config.SAVE_DIR, 'best_model.pth')
        torch.save(checkpoint, model_path)
        print(f"  💾 Saved best model (val_loss: {val_loss:.3f})")
    else:
        patience_counter += 1
        print(f"  ⏳ Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("\n⚠️  Early stopping triggered!")
            break

# -------------------------
# 11. Save Training History
# -------------------------
history_path = os.path.join(config.SAVE_DIR, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(history, f, indent=2)

# -------------------------
# 12. Test Set Evaluation
# -------------------------
print("\n📊 Evaluating on test set...")
checkpoint = torch.load(os.path.join(config.SAVE_DIR, 'best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = evaluate(model, test_loader, criterion)
test_word_acc = compute_word_accuracy(model, test_loader, idx2char, device)

print(f"\n🎉 Test Results:")
print(f"   Loss: {test_loss:.3f}")
print(f"   Char Accuracy: {test_acc:.3f}")
print(f"   Word Accuracy: {test_word_acc:.3f}")

# -------------------------
# 13. Prediction Function
# -------------------------
def predict(text, model, char2idx, idx2char, device, max_len=config.MAX_LEN):
    """Correct spelling of input text"""
    model.eval()
    with torch.no_grad():
        # Encode input
        chars = [SOS_TOKEN] + list(text)[:max_len-2] + [EOS_TOKEN]
        indices = [char2idx.get(c, char2idx[UNK_TOKEN]) for c in chars]
        src = torch.tensor(indices).unsqueeze(0).to(device)

        # Generate prediction
        output = model(src, trg=None, teacher_forcing_ratio=0.0)
        pred_seq = output.argmax(2).squeeze(0).cpu().numpy()

        return seq_to_text(pred_seq, idx2char)

# -------------------------
# 14. Sample Predictions
# -------------------------
print("\n📝 Sample Predictions:\n")
num_samples = 10
sample_indices = random.sample(range(len(test_data)), min(num_samples, len(test_data)))

for idx in sample_indices:
    item = test_data[idx]
    noisy = item.get("noisy", "")
    clean = item.get("clean", "")
    predicted = predict(noisy, model, char2idx, idx2char, device)

    match = "✓" if predicted == clean else "✗"
    print(f"{match} Noisy:     {noisy}")
    print(f"  Clean:     {clean}")
    print(f"  Predicted: {predicted}")
    print()

print(f"\n✅ All files saved to: {config.SAVE_DIR}")
print("\nFiles created:")
print(f"  - best_model.pth (model checkpoint)")
print(f"  - vocab.json (vocabulary)")
print(f"  - training_history.json (metrics)")