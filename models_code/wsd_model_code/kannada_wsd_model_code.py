"""
Kannada Word Sense Disambiguation (WSD) Model
Uses multilingual BERT optimized for Kannada text
WITH MODEL DOWNLOAD FEATURE
"""

# ========================================
# STEP 1: Install Required Libraries
# ========================================
!pip install transformers datasets torch scikit-learn pandas numpy -q

# ========================================
# STEP 2: Import Libraries
# ========================================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import warnings
import random
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ========================================
# STEP 3: Load Your Kannada Dataset
# ========================================
print("\n" + "="*60)
print("📂 Upload your Kannada CSV file")
print("="*60)

from google.colab import files
uploaded = files.upload()

filename = list(uploaded.keys())[0]

# Try multiple parsing methods to handle different CSV formats
df = None
parsing_methods = [
    {'sep': '\t', 'encoding': 'utf-8', 'on_bad_lines': 'skip', 'name': 'Tab-separated with UTF-8'},
    {'sep': ',', 'encoding': 'utf-8', 'on_bad_lines': 'skip', 'name': 'Comma-separated with UTF-8'},
    {'sep': '\t', 'encoding': 'utf-8', 'name': 'Tab-separated'},
    {'sep': ',', 'encoding': 'utf-8', 'name': 'Comma-separated'},
    {'encoding': 'utf-8', 'name': 'Auto-detect with UTF-8'},
]

for method in parsing_methods:
    try:
        method_name = method.pop('name')
        df = pd.read_csv(filename, **method)
        if len(df.columns) >= 4:
            print(f"✅ Loaded with {method_name}")
            break
        else:
            print(f"⚠️ {method_name} only found {len(df.columns)} column(s), trying next method...")
            df = None
    except Exception as e:
        continue

if df is None:
    print(f"❌ Error: Could not load CSV with any parsing method")
    print("\n💡 Please ensure your CSV:")
    print("  - Is properly formatted")
    print("  - Has columns: Context, Gloss, Label, Target Word")
    print("  - Uses UTF-8 encoding")
    print("  - Columns are separated by comma (,) or tab (\\t)")
    raise ValueError("Could not parse CSV file")

print(f"\n✅ Dataset loaded successfully!")
print(f"Total samples: {len(df)}")
print(f"\nColumn names found: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head(10))

# ========================================
# STEP 4: Clean Dataset
# ========================================
print("\n" + "="*60)
print("🧹 Cleaning dataset...")
print("="*60)

print(f"Rows before cleaning: {len(df)}")

required_columns = ['Context', 'Gloss', 'Label', 'Target Word']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"\n⚠️ Missing columns: {missing_columns}")
    print(f"Available columns: {list(df.columns)}")
    print("\n💡 Trying to auto-fix column names...")
    df.columns = df.columns.str.strip()

if len(df.columns) < 4:
    print(f"\n❌ Error: Expected 4 columns, found {len(df.columns)}")
    print("Please ensure your CSV has: Context, Gloss, Label, Target Word")
    raise ValueError("Incorrect CSV format")

df = df.dropna()
df = df[~df['Context'].astype(str).str.contains('---', na=False)]
df = df[~df['Gloss'].astype(str).str.contains('---', na=False)]
df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
df = df.dropna(subset=['Label'])
df['Label'] = df['Label'].astype(int)
df = df.reset_index(drop=True)

print(f"Rows after cleaning: {len(df)}")
print(f"\nLabel distribution:")
print(df['Label'].value_counts())

unique_words = df['Target Word'].nunique()
print(f"\nUnique ambiguous words: {unique_words}")

print(f"\nSample of cleaned data:")
print(df.head())

if len(df) < 100:
    print("\n⚠️ WARNING: Dataset is very small (< 100 samples).")
elif len(df) < 500:
    print("\n⚠️ WARNING: Dataset is small (< 500 samples).")
else:
    print(f"\n✅ Good dataset size: {len(df)} samples")

# ========================================
# STEP 5: Dataset Class for Kannada
# ========================================
class KannadaWSDDataset(Dataset):
    def __init__(self, contexts, glosses, labels, tokenizer, max_length=128):
        self.contexts = contexts
        self.glosses = glosses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = str(self.contexts[idx])
        gloss = str(self.glosses[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            context,
            gloss,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ========================================
# STEP 6: Custom Model with Dropout
# ========================================
class KannadaBERTForWSD(nn.Module):
    def __init__(self, model_name, dropout_rate=0.3):
        super(KannadaBERTForWSD, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        for param in self.bert.embeddings.parameters():
            param.requires_grad = False

        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 2)
        self.layer_norm = nn.LayerNorm(768)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.layer_norm(pooled_output)

        x = self.dropout1(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {'loss': loss, 'logits': logits}

# ========================================
# STEP 7: Initialize Kannada Model
# ========================================
print("\n" + "="*60)
print("🤖 Initializing Kannada BERT model...")
print("="*60)

MODEL_NAME = "google/muril-base-cased"
print(f"📚 Model: {MODEL_NAME}")
print("   (MuRIL = Multilingual Representations for Indian Languages)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = KannadaBERTForWSD(MODEL_NAME, dropout_rate=0.4)
model.to(device)

print("✅ Model initialized successfully!")
print("\n💡 Anti-overfitting features enabled:")
print("  ✓ 40% Dropout layers")
print("  ✓ Frozen embeddings")
print("  ✓ Layer normalization")
print("  ✓ Weight decay regularization")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

# ========================================
# STEP 8: Split Dataset
# ========================================
print("\n" + "="*60)
print("📊 Splitting dataset...")
print("="*60)

train_contexts, test_contexts, train_glosses, test_glosses, train_labels, test_labels = train_test_split(
    df['Context'].values,
    df['Gloss'].values,
    df['Label'].values,
    test_size=0.2,
    random_state=42,
    stratify=df['Label'].values
)

print(f"Training samples: {len(train_contexts)}")
print(f"Testing samples: {len(test_contexts)}")

train_dataset = KannadaWSDDataset(train_contexts, train_glosses, train_labels, tokenizer)
test_dataset = KannadaWSDDataset(test_contexts, test_glosses, test_labels, tokenizer)

BATCH_SIZE = 8
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========================================
# STEP 9: Training Setup
# ========================================
EPOCHS = 15
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    eps=1e-8,
    weight_decay=0.01
)

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(WARMUP_RATIO * total_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print("\n⚙️ Training configuration:")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Warmup steps: {warmup_steps}")
print(f"  Weight decay: 0.01")

# ========================================
# STEP 10: Training Functions
# ========================================
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    losses = []
    correct_predictions = 0

    progress_bar = tqdm(data_loader, desc='Training')

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs['loss']
        logits = outputs['logits']

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    predictions = []
    true_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs['loss']
            logits = outputs['logits']

            probs = torch.softmax(logits, dim=1)
            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (correct_predictions.double() / len(data_loader.dataset),
            np.mean(losses),
            predictions,
            true_labels,
            all_probs)

# ========================================
# STEP 11: Train the Model
# ========================================
print("\n" + "="*60)
print("🎓 Starting training...")
print("="*60)

best_accuracy = 0
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    print(f'\n📅 Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 60)

    train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    print(f'Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}')

    val_acc, val_loss, _, _, _ = eval_model(model, test_loader, device)
    print(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}')

    gap = train_acc.item() - val_acc.item()
    print(f'Overfitting Gap: {gap:.4f} ({"⚠️ High" if gap > 0.15 else "✓ Okay"})')

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'best_kannada_wsd_model.pt')
        print(f'✅ Best model saved! Accuracy: {best_accuracy:.4f}')
    else:
        patience_counter += 1
        print(f'No improvement. Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\n⚠️ Early stopping triggered after {epoch + 1} epochs')
            break

print("\n" + "="*60)
print("🎉 Training completed!")
print(f"Best validation accuracy: {best_accuracy:.4f}")
print("="*60)

# ========================================
# STEP 12: Final Evaluation
# ========================================
print("\n" + "="*60)
print("📈 Final evaluation on test set")
print("="*60)

model.load_state_dict(torch.load('best_kannada_wsd_model.pt'))
test_acc, test_loss, predictions, true_labels, probs = eval_model(model, test_loader, device)

print(f'\n📊 FINAL RESULTS:')
print(f'Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)')
print(f'Test Loss: {test_loss:.4f}')

print("\n📋 Classification Report:")
print(classification_report(true_labels, predictions,
                          target_names=['Wrong Sense (0)', 'Correct Sense (1)'],
                          digits=4))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(true_labels, predictions)
print(cm)
print(f"\nCorrect predictions: {cm[0][0] + cm[1][1]} / {len(true_labels)}")

avg_confidence = np.mean([max(p) for p in probs])
print(f"\n💪 Average confidence: {avg_confidence:.4f}")

# ========================================
# STEP 13: DOWNLOAD MODEL
# ========================================
print("\n" + "="*60)
print("💾 DOWNLOADING TRAINED MODEL")
print("="*60)

print("\n🔽 Downloading best_kannada_wsd_model.pt...")
files.download('best_kannada_wsd_model.pt')
print("✅ Model downloaded successfully!")

print("\n📦 To use this model later:")
print("  1. Upload the .pt file to Colab")
print("  2. Load with: model.load_state_dict(torch.load('best_kannada_wsd_model.pt'))")

# ========================================
# STEP 14: Prediction Function
# ========================================
def predict_kannada_word_sense(context, gloss, target_word):
    model.eval()

    encoding = tokenizer.encode_plus(
        context,
        gloss,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction, confidence

# ========================================
# STEP 15: Test with Examples
# ========================================
print("\n" + "="*60)
print("🧪 Testing with sample predictions")
print("="*60)

sample_contexts = df['Context'].unique()[:3]
print(f"\nTesting with {len(sample_contexts)} example contexts from your data:")

for context in sample_contexts:
    context_data = df[df['Context'] == context]
    target_word = context_data['Target Word'].iloc[0]

    print(f"\n📝 Context: '{context}'")
    print(f"🎯 Target word: '{target_word}'")
    print("\nTesting glosses:")

    for idx, row in context_data.iterrows():
        gloss = row['Gloss']
        true_label = row['Label']

        prediction, confidence = predict_kannada_word_sense(context, gloss, target_word)

        result = "✓ CORRECT SENSE" if prediction == 1 else "✗ WRONG SENSE"
        actual = "✓ Should be correct" if true_label == 1 else "✗ Should be wrong"
        match = "✅" if prediction == true_label else "❌"
        conf_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.6 else "🔴"

        print(f"\n  Gloss: '{gloss}'")
        print(f"  Prediction: {result}")
        print(f"  Actual: {actual}")
        print(f"  Match: {match}")
        print(f"  Confidence: {conf_emoji} {confidence:.4f}")

print("\n" + "="*60)
print("✅ Kannada WSD model training complete!")
print("="*60)
print("\n💡 Model has been downloaded to your computer!")
print("📁 File: best_kannada_wsd_model.pt")