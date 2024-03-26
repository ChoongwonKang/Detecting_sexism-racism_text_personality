import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Subset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import os

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    input_ids = []
    attention_masks = []
    for text in df['prepro_text_2']:
        encoded_dict = tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=64,
                            padding='max_length',
                            return_attention_mask=True,
                            return_tensors='pt',
                       )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['label'].values)

    dataset = TensorDataset(input_ids, attention_masks, labels)
    train_val_ids, test_ids, train_val_labels, test_labels = train_test_split(
        range(len(dataset)), labels.numpy(), test_size=0.2, stratify=labels.numpy(), random_state=42)
    train_ids, val_ids, _, _ = train_test_split(
        train_val_ids, train_val_labels, test_size=0.25, stratify=train_val_labels, random_state=42)

    train_dataset = Subset(dataset, train_ids)
    val_dataset = Subset(dataset, val_ids)
    test_dataset = Subset(dataset, test_ids)

    return train_dataset, val_dataset, test_dataset

def evaluate(model, dataloader, device):
    model.eval()
    eval_accuracy = 0
    nb_eval_steps = 0
    all_preds, all_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.inference_mode():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs.logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        preds = np.argmax(logits, axis=1).flatten()
        labels = label_ids.flatten()

        all_preds.extend(preds)
        all_labels.extend(labels)
        
        tmp_eval_accuracy = np.sum(preds == labels) / len(labels)
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

    final_accuracy = eval_accuracy / nb_eval_steps
    f1 = f1_score(all_labels, all_preds, average='binary')
    return final_accuracy, f1

def main(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset, test_dataset = load_and_preprocess_data(file_path)
    batch_size = 32

    train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs)

    best_val_accuracy = 0
    best_val_f1 = 0
    early_stopping_patience = 4
    early_stopping_counter = 0
    best_model_path = 'best_model.pth'

    for epoch_i in range(epochs):
        model.train()
        total_train_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch_i + 1}", total=len(train_dataloader)):
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()        
            result = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
            loss = result.loss
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        
        val_accuracy, val_f1 = evaluate(model, validation_dataloader, device)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(best_model_path))
    test_accuracy, test_f1 = evaluate(model, test_dataloader, device)
    print(f"Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    os.chdir('C:\\Users\\david\\Desktop\\대학원\\Individual_project\\mbti_project\\Hatespeech_data\\LIWC')
    file_path = 'data.csv'
    main(file_path)
