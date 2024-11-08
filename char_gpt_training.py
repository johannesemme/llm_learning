import torch
from char_gpt_dataloading import get_dataloader, Tokenizer
from char_gpt_model import GPTModel
import torch.nn.functional as F

if __name__ == "__main__": 
    
    torch.manual_seed(3333)
    
    # Parameters
    block_size = 32  # Length of each input sequence
    batch_size = 16   # Number of sequences per batch
    train_split = 0.9
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load text 
    with open("data/tinyshakespeare.txt", 'r', encoding='utf-8') as f:
        text = f.read()

    # Tokenize the data
    tokenizer = Tokenizer(text)
    #tokenizer.add_token("<|endoftext|>")
    encoded_text = tokenizer.encode(text)
    data = torch.tensor(encoded_text, dtype=torch.long)
    
    # Print vocabulary size
    print("Vocabulary size:", tokenizer.get_vocab_size())
    
    # Split into train and test data
    n = int(train_split*len(data)) 
    train_data = data[:n]
    val_data = data[n:]
    
    # Get dataloaders
    train_loader = get_dataloader(train_data, block_size, batch_size)
    val_loader = get_dataloader(val_data, block_size, batch_size)    
    
    # Model configuration
    model_cfg = {
        "vocab_size": tokenizer.get_vocab_size(),
        "context_length": block_size,
        "emb_dim": 64,
        "drop_rate": 0.0,
        "n_heads": 4,
        "n_layers": 4,
        "qkv_bias": False
    }
    model = GPTModel(model_cfg).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    max_iter = 10_000
    
    train_losses = []
    eval_losses = []
    current_best_loss = 1e9
    for epoch in range(1, 2):
        model.train()
        for i, (xb, yb) in enumerate(train_loader):
            x, y = xb.to(device), yb.to(device) # (B, T), (B, T)
            optimizer.zero_grad()
            logits = model(x) # (B, T, vocab_size)

            # Compute loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)) # logits: (B*T, vocab_size), y: (B*T)
        
            # Backward pass
            loss.backward()
            optimizer.step()

            
            if i % 100 == 0:
                # estimate train loss by 200 batches
                estimated_train_loss = 0
                for j, (xb, yb) in enumerate(train_loader):
                    if j == 200:
                        break
                    x, y = xb.to(device), yb.to(device)
                    with torch.no_grad():
                        logits = model(x)
                        train_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
                        estimated_train_loss += train_loss
                estimated_train_loss /= 200
                
                # estimate eval loss by 200 batches
                model.eval()
                estimated_eval_loss = 0
                for j, (xb, yb) in enumerate(val_loader):
                    if j == 200:
                        break
                    x, y = xb.to(device), yb.to(device)
                    with torch.no_grad():
                        logits = model(x)
                        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item()
                        estimated_eval_loss += val_loss
                estimated_eval_loss /= 200
                
                # log losses
                train_losses.append(estimated_train_loss)
                eval_losses.append(estimated_eval_loss)
                
                print(f"Epoch: {epoch}, step: {i}, train loss: {estimated_train_loss:.4f}, eval loss: {estimated_eval_loss:.4f}")
                
            if i % 1_000 == 0 and i > 0:               
                if estimated_eval_loss < current_best_loss:
                    current_best_loss = estimated_eval_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'cfg': model_cfg,
                        'tokenizer': tokenizer,
                        'epoch': epoch,
                        'step': i,
                    }, f'model_checkpoints/char_gpt/best_model.pt')
                
            if i > max_iter:
                exit()
        