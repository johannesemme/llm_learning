import torch
from char_gpt_model import GPTModel
import torch.nn.functional as F
import warnings

if __name__ == "__main__":
    device = "cpu"
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    model_path = 'model_checkpoints/char_gpt/best_model.pt'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        checkpoint = torch.load(model_path, weights_only=False)

    
    print(f"Best model load. It was found at epoch {checkpoint['epoch']} and step {checkpoint['step']}")

    model_cfg = checkpoint['cfg']
    tokenizer = checkpoint['tokenizer']

    model = GPTModel(model_cfg).to(device) # init model "skeleton"
    model.load_state_dict(checkpoint['model_state_dict']) # load model weights from checkpoint

    def generate_text_simple(model, idx, max_new_tokens, context_size):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -context_size:]
            # get the predictions
            logits = model(idx_cond)
            # focus only on the last token
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities of the last dim (the vocab size)
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the "running sequence"
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

    # Generate text
    max_new_tokens = 1_000
    start_text = "\n"

    encoded_text = tokenizer.encode(start_text)
    encoded_tensor = torch.tensor(encoded_text, dtype=torch.long).unsqueeze(0).to(device)

    out = generate_text_simple(model, encoded_tensor, max_new_tokens, model_cfg['context_length'])
    decode_text = tokenizer.decode(out.squeeze().tolist())

    print(decode_text)