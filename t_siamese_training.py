import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BigBirdModel, BigBirdTokenizer
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Define the Triplet Loss with Cosine Similarity
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, anchor, positive, negative):
        positive_distance = 1 - self.cos_sim(anchor, positive)
        negative_distance = 1 - self.cos_sim(anchor, negative)
        loss = torch.mean(torch.relu(positive_distance - negative_distance + self.margin))
        return loss

# Load the dataset
file_path = 'sampled_dataset.csv'  # Update with the correct path
df = pd.read_csv(file_path)[:10]

# Construct triplets (reference answer, ChatGPT answer, human answer)
triplets = []

for _, row in df.iterrows():
    reference_answer = row['ReferenceAnswer']
    human_answer = row['HumanAnswer']
    chatgpt_answer = row['ChatGPTAnswer']
    
    triplets.append((reference_answer, chatgpt_answer, human_answer))

# Create a DataFrame from the triplets
triplets_df = pd.DataFrame(triplets, columns=['ReferenceAnswer', 'ChatGPTAnswer', 'HumanAnswer'])

# Define the dataset class for triplets
class QADatasetTriplet(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        anchor = row['ReferenceAnswer']
        positive = row['ChatGPTAnswer']
        negative = row['HumanAnswer']
        
        anchor_inputs = self.tokenizer.encode_plus(
            anchor, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True
        )
        
        positive_inputs = self.tokenizer.encode_plus(
            positive, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True
        )
        
        negative_inputs = self.tokenizer.encode_plus(
            negative, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            return_token_type_ids=True, 
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True
        )

        return {
            'anchor_input_ids': torch.tensor(anchor_inputs['input_ids'], dtype=torch.long),
            'anchor_attention_mask': torch.tensor(anchor_inputs['attention_mask'], dtype=torch.long),
            'anchor_token_type_ids': torch.tensor(anchor_inputs['token_type_ids'], dtype=torch.long),
            'positive_input_ids': torch.tensor(positive_inputs['input_ids'], dtype=torch.long),
            'positive_attention_mask': torch.tensor(positive_inputs['attention_mask'], dtype=torch.long),
            'positive_token_type_ids': torch.tensor(positive_inputs['token_type_ids'], dtype=torch.long),
            'negative_input_ids': torch.tensor(negative_inputs['input_ids'], dtype=torch.long),
            'negative_attention_mask': torch.tensor(negative_inputs['attention_mask'], dtype=torch.long),
            'negative_token_type_ids': torch.tensor(negative_inputs['token_type_ids'], dtype=torch.long),
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
dataset = QADatasetTriplet(triplets_df, tokenizer, max_len)

# Split the dataset into training, validation, and test sets without random splitting
train_size = int(0.7 * len(triplets_df))
val_size = int(0.1 * len(triplets_df))
test_size = len(triplets_df) - train_size - val_size

train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size)))
test_dataset = torch.utils.data.Subset(dataset, list(range(train_size + val_size, len(triplets_df))))

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Initialize the model
model = BigBirdSiameseNetwork('google/bigbird-roberta-base')

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = TripletLoss(margin=0.6)

# Early Stopping parameters
patience = 3
best_val_loss = float('inf')
epochs_no_improve = 0

# Open a file to record the validation results
val_results_file = open("validation_results.txt", "w")
val_results=[]
# Training loop with early stopping
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        anchor_input_ids = batch['anchor_input_ids'].to(device)
        anchor_attention_mask = batch['anchor_attention_mask'].to(device)
        anchor_token_type_ids = batch['anchor_token_type_ids'].to(device)
        
        positive_input_ids = batch['positive_input_ids'].to(device)
        positive_attention_mask = batch['positive_attention_mask'].to(device)
        positive_token_type_ids = batch['positive_token_type_ids'].to(device)
        
        negative_input_ids = batch['negative_input_ids'].to(device)
        negative_attention_mask = batch['negative_attention_mask'].to(device)
        negative_token_type_ids = batch['negative_token_type_ids'].to(device)
        
        optimizer.zero_grad()
        anchor_output = model(anchor_input_ids, anchor_attention_mask, anchor_token_type_ids)
        positive_output = model(positive_input_ids, positive_attention_mask, positive_token_type_ids)
        negative_output = model(negative_input_ids, negative_attention_mask, negative_token_type_ids)
        
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Evaluation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            anchor_input_ids = batch['anchor_input_ids'].to(device)
            anchor_attention_mask = batch['anchor_attention_mask'].to(device)
            anchor_token_type_ids = batch['anchor_token_type_ids'].to(device)
            
            positive_input_ids = batch['positive_input_ids'].to(device)
            positive_attention_mask = batch['positive_attention_mask'].to(device)
            positive_token_type_ids = batch['positive_token_type_ids'].to(device)
            
            negative_input_ids = batch['negative_input_ids'].to(device)
            negative_attention_mask = batch['negative_attention_mask'].to(device)
            negative_token_type_ids = batch['negative_token_type_ids'].to(device)
            
            anchor_output = model(anchor_input_ids, anchor_attention_mask, anchor_token_type_ids)
            positive_output = model(positive_input_ids, positive_attention_mask, positive_token_type_ids)
            negative_output = model(negative_input_ids, negative_attention_mask, negative_token_type_ids)
            
            loss = criterion(anchor_output, positive_output, negative_output)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    # val_results.append((epoch + 1, val_loss))
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

    # Record the validation results
    val_results_file.write(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}\n')

    # Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break

# Close the file after training
val_results_file.close
