import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        # The number of samples we can create from the dataset
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Slice a sequence of length `block_size` for x and shift by one for y
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
    def add_token(self, token):
        if token not in self.chars:
            self.chars.append(token)
            self.vocab_size += 1
            self.stoi[token] = self.vocab_size - 1
            self.itos[self.vocab_size - 1] = token
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])
    
    def get_vocab_size(self):
        return self.vocab_size
    
def get_dataloader(data, block_size, batch_size, shuffle=True):
    dataset = CharDataset(data, block_size)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader
    
    
if __name__ == "__main__":
    # Parameters
    block_size = 128  # Length of each input sequence
    batch_size = 32   # Number of sequences per batch
    train_split = 0.9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load text 
    with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the data
    tokenizer = Tokenizer(text)
    encoded_text = tokenizer.encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)
    
    # Print vocabulary size
    print("Vocabulary size:", tokenizer.get_vocab_size())
    
    # Split into train and test data
    n = int(train_split*len(data)) 
    train_data = data[:n]
    val_data = data[n:]
    
    # Get dataloaders
    train_loader = get_dataloader(train_data, block_size, batch_size, shuffle=False)
    val_loader = get_dataloader(val_data, block_size, batch_size)    
    
    # Test data loaders
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # Move batch to the GPU if available
        print(x)
        
        for row in x:
            print(tokenizer.decode(row.tolist()))
            print("----"*10)
        
        print("Input shape:", x.shape)
        print("Output shape:", y.shape)
        break  
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)  # Move batch to the GPU if available
        print("Input shape:", x.shape)
        print("Output shape:", y.shape)
        break  