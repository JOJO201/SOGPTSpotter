import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BigBirdModel, BigBirdTokenizer
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

# Load the dataset
file_path = 'sampled_dataset.csv'  # Update with the correct path
df = pd.read_csv(file_path)[:10]

# Construct pairs (reference answer, human answer) and (reference answer, chatgpt answer)
pairs = []

for _, row in df.iterrows():
    reference_answer = row['ReferenceAnswer']
    human_answer = row['HumanAnswer']
    chatgpt_answer = row['ChatGPTAnswer']
    
    # Append (reference answer, human answer) with label 0 (no)
    pairs.append((reference_answer, human_answer, 0))
    
    # Append (reference answer, chatgpt answer) with label 1 (yes)
    pairs.append((reference_answer, chatgpt_answer, 1))

# Create a DataFrame from the sampled pairs
pairs_df = pd.DataFrame(pairs, columns=['ReferenceAnswer', 'Answer', 'Label'])

# Define the dataset class
class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text1 = row['ReferenceAnswer']
        text2 = row['Answer']
        label = row['Label']
        
        inputs1 = self.tokenizer.encode_plus(
            text1, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True
        )
        
        inputs2 = self.tokenizer.encode_plus(
            text2, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True
        )

        return {
            'input_ids1': torch.tensor(inputs1['input_ids'], dtype=torch.long),
            'attention_mask1': torch.tensor(inputs1['attention_mask'], dtype=torch.long),
            'token_type_ids1': torch.tensor(inputs1['token_type_ids'], dtype=torch.long),
            'input_ids2': torch.tensor(inputs2['input_ids'], dtype=torch.long),
            'attention_mask2': torch.tensor(inputs2['attention_mask'], dtype=torch.long),
            'token_type_ids2': torch.tensor(inputs2['token_type_ids'], dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Define the Siamese network using BigBird
class BigBirdSiameseNetwork(nn.Module):
    def __init__(self, bigbird_model_name):
        super(BigBirdSiameseNetwork, self).__init__()
        self.bigbird = BigBirdModel.from_pretrained(bigbird_model_name)
        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bigbird(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return pooled_output

# Initialize tokenizer and dataset
max_len = 1000  # Set max_len based on BigBird's capabilities
tokenizer = BigBirdTokenizer.from_pretrained('google/bigbird-roberta-base')
dataset = QADataset(pairs_df, tokenizer, max_len)

# Split the dataset into training, validation, and test sets without random splitting
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size)))
test_dataset = torch.utils.data.Subset(dataset, list(range(train_size + val_size, len(dataset))))

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Initialize the model
model = BigBirdSiameseNetwork('google/bigbird-roberta-base')

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation
model.eval()
test_preds = []
test_labels = []
cos_sim = nn.CosineSimilarity(dim=1)
with torch.no_grad():
    for batch in test_dataloader:
        input_ids1 = batch['input_ids1'].to(device)
        attention_mask1 = batch['attention_mask1'].to(device)
        token_type_ids1 = batch['token_type_ids1'].to(device)
        
        input_ids2 = batch['input_ids2'].to(device)
        attention_mask2 = batch['attention_mask2'].to(device)
        token_type_ids2 = batch['token_type_ids2'].to(device)
        
        labels = batch['label'].to(device)
        
        output1 = model(input_ids1, attention_mask1, token_type_ids1)
        output2 = model(input_ids2, attention_mask2, token_type_ids2)
        
        similarity = cos_sim(output1, output2)
        test_preds.extend(similarity.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Convert similarity scores to binary predictions
threshold = 0.5
binary_preds = [1 if p > threshold else 0 for p in test_preds]

# Evaluation metrics
f1 = f1_score(test_labels, binary_preds)
accuracy = accuracy_score(test_labels, binary_preds)
precision = precision_score(test_labels, binary_preds)
recall = recall_score(test_labels, binary_preds)
conf_matrix = confusion_matrix(test_labels, binary_preds)

# Save evaluation results to a file
with open("evaluation_results.txt", "w") as file:
    file.write(f'Test F1 Score: {f1:.4f}\n')
    file.write(f'Test Accuracy: {accuracy:.4f}\n')
    file.write(f'Test Precision: {precision:.4f}\n')
    file.write(f'Test Recall: {recall:.4f}\n')
    file.write(f'Confusion Matrix:\n{conf_matrix}\n')

print(f'Test F1 Score: {f1:.4f}')
print(f'Test Accuracy: {accuracy:.4f}')
print(f'Test Precision: {precision:.4f}')
print(f'Test Recall: {recall:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')
