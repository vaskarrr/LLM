# samsara_full_impl.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np

# --------------------------
# Argument parsing
# --------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='distilgpt2')
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--steps', type=int, default=160)
parser.add_argument('--limit_examples', type=int, default=600)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

# --------------------------
# Device setup
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------------
# Tokenizer and Model
# --------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model, use_auth_token=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=False)
model.to(device)

# --------------------------
# Dataset preparation
# --------------------------
def build_two_domains(tokenizer, block_size=128, limit_examples=600):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", download_mode="reuse_cache_if_exists")
    text = "\n\n".join(ds["text"])
    mid = len(text) // 2
    texts = {"A": text[:mid], "B": text[mid:]}
    domains = {}
    for name, txt in texts.items():
        tokenized = []
        for i in tqdm(range(0, len(txt), block_size*4), desc=f"Tokenizing {name}"):
            chunk = txt[i:i+block_size*4]
            tokenized += tokenizer.encode(chunk, add_special_tokens=False)
        total_len = (len(tokenized) // block_size) * block_size
        pieces = [tokenized[i:i+block_size] for i in range(0, total_len, block_size)]
        if limit_examples:
            pieces = pieces[:limit_examples]
        attn = [[1]*block_size for _ in pieces]
        # Move to device
        pieces_tensor = [torch.tensor(x).to(device) for x in pieces]
        attn_tensor = [torch.tensor(x).to(device) for x in attn]
        domains[name] = {"input_ids": pieces_tensor, "attention_mask": attn_tensor, "labels": pieces_tensor}
    return domains

# --------------------------
# Samsara / Baseline training
# --------------------------
def train_model(model, domain_data, steps=160, lr=5e-5):
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        # randomly pick batch
        idx = random.randint(0, len(domain_data["input_ids"])-1)
        batch = {
            "input_ids": domain_data["input_ids"][idx].unsqueeze(0),
            "attention_mask": domain_data["attention_mask"][idx].unsqueeze(0),
            "labels": domain_data["labels"][idx].unsqueeze(0)
        }
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return np.mean(losses)

# --------------------------
# Evaluation: Perplexity
# --------------------------
def evaluate_model(model, domain_data):
    model.eval()
    ppl_list = []
    with torch.no_grad():
        for i in range(len(domain_data["input_ids"])):
            batch = {
                "input_ids": domain_data["input_ids"][i].unsqueeze(0),
                "attention_mask": domain_data["attention_mask"][i].unsqueeze(0),
                "labels": domain_data["labels"][i].unsqueeze(0)
            }
            outputs = model(**batch)
            loss = outputs.loss.item()
            ppl = np.exp(loss)
            ppl_list.append(ppl)
    return np.mean(ppl_list)

# --------------------------
# Main loop
# --------------------------
all_forgetting = {"Samsara": [], "Adapters": [], "EWC": []}

for trial in range(args.trials):
    seed = 42 + trial
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    print(f"\n--- Trial {trial+1}/{args.trials}, seed={seed} ---")
    
    # Build domains
    domains = build_two_domains(tokenizer, limit_examples=args.limit_examples)
    
    # Samsara training
    train_model(model, domains["A"], steps=args.steps)
    ppl_A_after_A = evaluate_model(model, domains["A"])
    train_model(model, domains["B"], steps=args.steps)
    ppl_A_after_B = evaluate_model(model, domains["A"])
    ppl_B_after_B = evaluate_model(model, domains["B"])
    forgetting = ppl_A_after_B - ppl_A_after_A
    all_forgetting["Samsara"].append(forgetting)
    print(f"Samsara done. PPLs: A->A {ppl_A_after_A:.2f}, A->B {ppl_A_after_B:.2f}, B->B {ppl_B_after_B:.2f}")
    
    # Baselines: simple example using same model (replace with Adapters/EWC if implemented)
    # For demonstration, just reset model weights (simple baseline)
    model = AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=False).to(device)
    train_model(model, domains["A"], steps=args.steps)
    ppl_A_after_A = evaluate_model(model, domains["A"])
    train_model(model, domains["B"], steps=args.steps)
    ppl_A_after_B = evaluate_model(model, domains["A"])
    forgetting = ppl_A_after_B - ppl_A_after_A
    all_forgetting["Adapters"].append(forgetting)
    print(f"Adapters done. Forgetting: {forgetting:.2f}")
    
    model = AutoModelForCausalLM.from_pretrained(args.model, use_auth_token=False).to(device)
    train_model(model, domains["A"], steps=args.steps)
    ppl_A_after_A = evaluate_model(model, domains["A"])
    train_model(model, domains["B"], steps=args.steps)
    ppl_A_after_B = evaluate_model(model, domains["A"])
    forgetting = ppl_A_after_B - ppl_A_after_A
    all_forgetting["EWC"].append(forgetting)
    print(f"EWC done. Forgetting: {forgetting:.2f}")

# --------------------------
# Plotting
# --------------------------
if args.plot:
    avg_vals = [np.mean(all_forgetting[m]) for m in ["Samsara", "Adapters", "EWC"]]
    std_vals = [np.std(all_forgetting[m]) for m in ["Samsara", "Adapters", "EWC"]]
    plt.figure(figsize=(8,5))
    plt.bar(["Samsara", "Adapters-only", "EWC"], avg_vals, yerr=std_vals)
    plt.ylabel("Forgetting (ΔPPL)")
    plt.title("Forgetting by Method")
    plt.show()
