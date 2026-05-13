# =========================
# Kannada Spelling Correction Seq2Seq (Production Ready)
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

# -------------------------
# 0. Configuration
# -------------------------
class Config:
    # Paths
    DRIVE_PATH = "/content/drive/MyDrive/Project DD using ML /dataset/spelling dataset/kn_noisy_pairs.jsonl"
    SAVE_DIR = "/content/drive/MyDrive/kannada_model"

    # Data splits
    TRAIN_SIZE = 150_000
    VAL_SIZE = 10_000
    TEST_SIZE = 5_000
    MAX_LEN = 64  # Reduced from 128 for Kannada
    MIN_FREQ = 2  # Minimum character frequency

    # Model architecture
    EMB_DIM = 256
    HID_DIM = 512
    DROPOUT = 0.3

    # Training hyperparameters
    BATCH_SIZE = 128
    N_EPOCHS = 10
    LR = 1e-3
    GRAD_CLIP = 1.0
    PATIENCE = 3

    # Teacher forcing schedule (decay from 1.0 to 0.5)
    TF_START = 1.0
    TF_END = 0.5

    # Misc
    SEED = 42
    NUM_WORKERS = 2

config = Config()

# -------------------------
# 1. Setup & Mount Drive
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

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)
os.makedirs(config.SAVE_DIR, exist_ok=True)

if not os.path.exists(config.DRIVE_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at: {config.DRIVE_PATH}")
print(f"✅ Found Kannada dataset at: {config.DRIVE_PATH}")

# -------------------------
# 2. Load and Split Data (NO OVERLAP!)
# -------------------------
print("\n📊 Loading Kannada dataset with proper splits...")
dataset = load_dataset("json", data_files=config.DRIVE_PATH, split="train", streaming=True)

# Shuffle and create non-overlapping splits
dataset_shuffled = dataset.shuffle(seed=config.SEED, buffer_size=10000)

print("Loading validation set...")
val_data = list(dataset_shuffled.take(config.VAL_SIZE))

print("Loading test set...")
test_iter = dataset_shuffled.skip(config.VAL_SIZE)
test_data = list(test_iter.take(config.TEST_SIZE))

print("Loading training set...")
train_iter = dataset_shuffled.skip(config.VAL_SIZE + config.TEST_SIZE)
train_data = list(train_iter.take(config.TRAIN_SIZE))

print(f"✅ Splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")

# -------------------------
# 3. Build Kannada Vocabulary (from training only)
# -------------------------
PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<PAD>", "<SOS>", "<EOS>", "<UNK>"

print("\n📖 Building Kannada character vocabulary from training data...")
char_counter = Counter()
max_word_len = 0

for item in train_data:
    for key in ["noisy", "clean"]:
        text = item.get(key, "")
        if text:
            char_counter.update(list(text))
            max_word_len = max(max_word_len, len(text))

# Filter characters by frequency
frequent_chars = [ch for ch, cnt in char_counter.items() if cnt >= config.MIN_FREQ]
char_list = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + sorted(frequent_chars)

char2idx = {ch: i for i, ch in enumerate(char_list)}
idx2char = {i: ch for ch, i in char2idx.items()}
VOCAB_SIZE = len(char2idx)

print(f"✅ Kannada vocabulary size: {VOCAB_SIZE}")
print(f"   Total unique characters: {len(frequent_chars)}")
print(f"   Max word length in training: {max_word_len}")
print(f"   Top 10 characters: {char_counter.most_common(10)}")

# Save vocabulary
vocab_path = os.path.join(config.SAVE_DIR, "kannada_vocab.json")
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump({
        "char2idx": char2idx,
        "idx2char": idx2char,
        "vocab_size": VOCAB_SIZE,
        "max_word_len": max_word_len
    }, f, ensure_ascii=False, indent=2)
print(f"💾 Vocabulary saved to: {vocab_path}")

# -------------------------
# 4. Dataset with Dynamic Padding
# -------------------------
class KannadaSpellingDataset(Dataset):
    def __init__(self, items, char2idx, max_len=config.MAX_LEN):
        self.items = items
        self.char2idx = char2idx
        self.max_len = max_len
        self.pad_idx = char2idx[PAD_TOKEN]
        self.sos_idx = char2idx[SOS_TOKEN]
        self.eos_idx = char2idx[EOS_TOKEN]
        self.unk_idx = char2idx[UNK_TOKEN]

    def encode(self, text):
        """Encode text to indices with SOS and EOS"""
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

# Create datasets and dataloaders
train_dataset = KannadaSpellingDataset(train_data, char2idx)
val_dataset = KannadaSpellingDataset(val_data, char2idx)
test_dataset = KannadaSpellingDataset(test_data, char2idx)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                         shuffle=True, collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                       shuffle=False, collate_fn=collate_fn, num_workers=config.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                        shuffle=False, collate_fn=collate_fn, num_workers=config.NUM_WORKERS)

print(f"✅ DataLoaders created (batch_size={config.BATCH_SIZE})")

# -------------------------
# 5. Model Architecture
# -------------------------
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=char2idx[PAD_TOKEN])
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True,
                          bidirectional=True, num_layers=1)
        self.fc_h = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_c = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        # Combine bidirectional states
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
print("\n🏗️  Building Kannada Seq2Seq model...")
encoder = Encoder(VOCAB_SIZE, config.EMB_DIM, config.HID_DIM, config.DROPOUT)
decoder = Decoder(VOCAB_SIZE, config.EMB_DIM, config.HID_DIM,
                 config.HID_DIM * 2, config.DROPOUT)
model = Seq2Seq(encoder, decoder, device).to(device)

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
# 8. Metrics & Evaluation
# -------------------------
def compute_char_accuracy(preds, targets):
    """Character-level accuracy (ignoring padding)"""
    preds = preds.argmax(2)
    mask = targets != char2idx[PAD_TOKEN]
    correct = ((preds == targets) & mask).sum().item()
    total = mask.sum().item()
    return correct / total if total > 0 else 0

def seq_to_text(seq, idx2char):
    """Convert sequence of indices to Kannada text"""
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
    """Word-level accuracy - full word must be correct"""
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
# 9. Training Functions
# -------------------------
def train_epoch(model, loader, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    progress_bar = tqdm(loader, desc="Training", leave=False)
    for src, trg in progress_bar:
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output_flat = output[:, 1:].reshape(-1, output_dim)
        trg_flat = trg[:, 1:].reshape(-1)

        loss = criterion(output_flat, trg_flat)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += compute_char_accuracy(output[:, 1:], trg[:, 1:])

        progress_bar.set_postfix({'loss': f'{loss.item():.3f}'})

    return epoch_loss / len(loader), epoch_acc / len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for src, trg in tqdm(loader, desc="Evaluating", leave=False):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0.0)

            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)

            loss = criterion(output_flat, trg_flat)

            epoch_loss += loss.item()
            epoch_acc += compute_char_accuracy(output[:, 1:], trg[:, 1:])

    return epoch_loss / len(loader), epoch_acc / len(loader)

# -------------------------
# 10. Training Loop with Early Stopping
# -------------------------
print("\n🎯 Starting training for Kannada spelling correction...\n")

best_val_loss = float('inf')
patience_counter = 0
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': [],
    'word_acc': [], 'learning_rates': []
}

for epoch in range(config.N_EPOCHS):
    # Calculate teacher forcing ratio (linear decay)
    tf_ratio = config.TF_START - (config.TF_START - config.TF_END) * (epoch / config.N_EPOCHS)

    start_time = time.time()

    train_loss, train_acc = train_epoch(model, train_loader, optimizer,
                                       criterion, config.GRAD_CLIP, tf_ratio)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    word_acc = compute_word_accuracy(model, val_loader, idx2char, device)

    end_time = time.time()
    epoch_mins = (end_time - start_time) / 60

    # Update learning rate scheduler
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['word_acc'].append(word_acc)
    history['learning_rates'].append(current_lr)

    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{config.N_EPOCHS} | Time: {epoch_mins:.1f}m | TF: {tf_ratio:.2f} | LR: {current_lr:.2e}")
    print(f"{'='*60}")
    print(f"  Train Loss: {train_loss:.4f} | Train Char Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Char Acc:   {val_acc:.4f}")
    print(f"  Val Word Accuracy: {word_acc:.4f} ({word_acc*100:.2f}%)")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'word_acc': word_acc,
            'config': vars(config),
            'vocab_size': VOCAB_SIZE
        }

        model_path = os.path.join(config.SAVE_DIR, 'kannada_best_model.pth')
        torch.save(checkpoint, model_path)
        print(f"  💾 ✨ New best model saved! (val_loss: {val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  ⏳ No improvement. Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("\n⚠️  Early stopping triggered - no improvement for 3 epochs!")
            break

print(f"\n{'='*60}")
print("Training completed!")
print(f"{'='*60}")

# -------------------------
# 11. Save Training History
# -------------------------
history_path = os.path.join(config.SAVE_DIR, 'kannada_training_history.json')
with open(history_path, 'w', encoding='utf-8') as f:
    json.dump(history, f, indent=2, ensure_ascii=False)
print(f"📊 Training history saved to: {history_path}")

# -------------------------
# 12. Test Set Evaluation
# -------------------------
print(f"\n{'='*60}")
print("📊 Final Evaluation on Test Set")
print(f"{'='*60}")

# Load best model
checkpoint = torch.load(os.path.join(config.SAVE_DIR, 'kannada_best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ Loaded best model from epoch {checkpoint['epoch']+1}")

test_loss, test_char_acc = evaluate(model, test_loader, criterion)
test_word_acc = compute_word_accuracy(model, test_loader, idx2char, device)

print(f"\n🎉 Test Set Results:")
print(f"   Loss:           {test_loss:.4f}")
print(f"   Char Accuracy:  {test_char_acc:.4f} ({test_char_acc*100:.2f}%)")
print(f"   Word Accuracy:  {test_word_acc:.4f} ({test_word_acc*100:.2f}%)")

# -------------------------
# 13. Prediction Function
# -------------------------
def predict_kannada(text, model, char2idx, idx2char, device, max_len=config.MAX_LEN):
    """Correct spelling of Kannada text"""
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
print(f"\n{'='*60}")
print("📝 Sample Predictions on Test Set")
print(f"{'='*60}\n")

num_samples = 15
sample_indices = random.sample(range(len(test_data)), min(num_samples, len(test_data)))

correct_count = 0
for i, idx in enumerate(sample_indices, 1):
    item = test_data[idx]
    noisy = item.get("noisy", "")
    clean = item.get("clean", "")
    predicted = predict_kannada(noisy, model, char2idx, idx2char, device)

    is_correct = predicted == clean
    if is_correct:
        correct_count += 1

    status = "✅" if is_correct else "❌"
    print(f"{status} Sample {i}:")
    print(f"   Noisy:     {noisy}")
    print(f"   Ground Truth: {clean}")
    print(f"   Predicted:    {predicted}")
    if not is_correct:
        print(f"   Difference: GT≠Pred")
    print()

print(f"Sample accuracy: {correct_count}/{num_samples} ({correct_count/num_samples*100:.1f}%)\n")

# -------------------------
# 15. Summary
# -------------------------
print(f"{'='*60}")
print("✅ All files saved successfully!")
print(f"{'='*60}")
print(f"\nSaved files in: {config.SAVE_DIR}")
print(f"  📦 kannada_best_model.pth     - Model checkpoint")
print(f"  📖 kannada_vocab.json         - Vocabulary mapping")
print(f"  📊 kannada_training_history.json - Training metrics")
print(f"\n{'='*60}")
print("🎉 Kannada Spelling Correction Model Training Complete!")
print(f"{'='*60}\n")

# -------------------------
# 16. Quick Inference Example
# -------------------------
print("💡 Quick inference example:")
print("-" * 60)
example_noisy = test_data[0].get("noisy", "")
example_corrected = predict_kannada(example_noisy, model, char2idx, idx2char, device)
print(f"Input:  {example_noisy}")
print(f"Output: {example_corrected}")
print("-" * 60)