#!/usr/bin/env python3
"""
samsara_full_impl_polished.py

Polished, research-demo script for "Samsara Cycles" with:
 - Samsara Cycle (LoRA Soul + cremation + rebirth + karmic alignment)
 - Adapters-only baseline
 - EWC baseline
 - Multi-seed trials, mean/std aggregation
 - Optional plots and paired t-tests (requires scipy & matplotlib)

Important: to silence a common LoRA warning when using GPT-like Conv1D modules,
we set fan_in_fan_out=True in the LoraConfig.

Usage (small demo):
  pip install -U torch transformers datasets peft matplotlib scipy tqdm
  python samsara_full_impl_polished.py --model distilgpt2 --trials 3 --steps 160 --limit_examples 600 --plot

For larger / paper runs: increase --steps and --limit_examples and use a larger model.
"""

import argparse
import copy
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict

import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

# Optional plotting & stats
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
try:
    from scipy import stats
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# -------------------------
# Utilities
# -------------------------
def set_seed(s: int):
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def perp_from_loss(loss):
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")

def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}

# -------------------------
# Dataset helpers
# -------------------------
def build_two_domains(tokenizer, block_size=128, limit_examples=800):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    text = "\n\n".join(ds["train"]["text"])
    mid = len(text) // 2
    texts = {"A": text[:mid], "B": text[mid:]}
    domains = {}
    for name, txt in texts.items():
        tokenized = tokenizer(txt)["input_ids"]
        total_len = (len(tokenized)//block_size)*block_size
        pieces = [tokenized[i:i+block_size] for i in range(0, total_len, block_size)]
        if limit_examples:
            pieces = pieces[:limit_examples]
        attn = [[1]*block_size for _ in pieces]
        domains[name] = {"input_ids": pieces, "attention_mask": attn, "labels": pieces}
    return domains

class DictDataset(torch.utils.data.Dataset):
    def __init__(self, d): self.d = d; self.n = len(self.d["input_ids"])
    def __len__(self): return self.n
    def __getitem__(self, i): return {k: torch.tensor(self.d[k][i]) for k in self.d.keys()}

# -------------------------
# Soul + triggers (LoRA)
# -------------------------
def attach_lora(model, r=8, alpha=16, dropout=0.05, target_modules=("c_attn","c_proj")):
    # NOTE: fan_in_fan_out=True prevents the PEFT warning with Conv1D (GPT-style)
    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=True  # explicit to avoid Conv1D warning
    )
    return get_peft_model(model, cfg)

def extract_soul(model: PeftModel):
    soul = {}
    for name, p in model.named_parameters():
        if "lora_" in name and p.requires_grad:
            soul[name] = p.detach().cpu().clone()
    return soul

def load_soul(model: PeftModel, soul: dict):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in soul:
                p.copy_(soul[name].to(p.device))

def build_trigger_tokens(tokenizer):
    triggers = set()
    for tok, idx in tokenizer.get_vocab().items():
        if any(ch.isdigit() for ch in tok) or tok in ['.', ',', ':', ';', '(', ')', '[', ']', '"', "'"]:
            triggers.add(idx)
    return triggers

def karmic_alignment_loss(hidden_states, input_ids, trigger_set, weight=0.01):
    if not trigger_set or weight == 0.0:
        return torch.tensor(0.0, device=hidden_states.device)
    B, T, H = hidden_states.shape
    mask = torch.zeros(B, T, dtype=torch.bool, device=hidden_states.device)
    ids = input_ids
    for b in range(B):
        for t in range(T):
            if int(ids[b, t].item()) in trigger_set:
                mask[b, t] = True
    if mask.sum() == 0:
        return torch.tensor(0.0, device=hidden_states.device)
    feats = hidden_states[mask]
    mean_vec = feats.mean(dim=0, keepdim=True)
    return weight * ((feats - mean_vec) ** 2).mean()

# -------------------------
# Cremation: random or Fisher importance
# -------------------------
def select_cremation_names_random(model: nn.Module, pct: float, seed: int):
    rng = random.Random(seed)
    candidates = []
    for n, p in model.named_parameters():
        ln = n.lower()
        if any(k in ln for k in ["embed", "wpe", "wte", "ln_", "layernorm", "bias", "norm"]):
            continue
        if "lora_" in ln:
            continue
        if p.requires_grad and p.dim() > 1:
            candidates.append(n)
    rng.shuffle(candidates)
    k = max(1, int(len(candidates)*pct))
    return set(candidates[:k])

def compute_diagonal_fisher(model: nn.Module, tokenizer, dataset_dict, device, max_samples=64, bsz=8):
    model.train()
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = torch.zeros_like(p, device=device)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    ds = DictDataset(dataset_dict)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True)
    seen = 0
    for batch in dl:
        batch = to_device(batch, device)
        outputs = model(**batch)
        logits = outputs.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()
        loss = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        model.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.detach() ** 2
        seen += batch["input_ids"].size(0)
        if seen >= max_samples:
            break
    for n in fisher:
        fisher[n] /= max(1, seen)
    return fisher

def select_cremation_names_fisher(fisher_dict: dict, pct: float):
    flat = []
    for n, mat in fisher_dict.items():
        flat.append((n, float(mat.mean().item()) if mat.numel()>0 else 0.0))
    flat = [x for x in flat if "lora_" not in x[0]]
    flat.sort(key=lambda x: x[1])
    k = max(1, int(len(flat)*pct))
    chosen = {name for name,_ in flat[:k]}
    return chosen

def reinit_params_by_name(module: nn.Module, names_to_reset: set):
    for n, p in module.named_parameters():
        if n in names_to_reset:
            if p.dim() > 1:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            else:
                nn.init.uniform_(p, -0.01, 0.01)

# -------------------------
# Train / eval loops
# -------------------------
def run_train_steps(model, tokenizer, domain, device, steps=200, bsz=8, lr=5e-5, warmup=10, trigger_set=None, karmic_w=0.01):
    model.train()
    ds = DictDataset(domain)
    dl = DataLoader(ds, batch_size=bsz, shuffle=True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=steps)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    step = 0
    while step < steps:
        for batch in dl:
            batch = to_device(batch, device)
            outputs = model(**batch, output_hidden_states=True)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            loss_main = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            hidden = outputs.hidden_states[-1]
            k_loss = karmic_alignment_loss(hidden, batch["input_ids"], trigger_set, weight=karmic_w)
            loss = loss_main + k_loss
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()
            step += 1
            if step >= steps:
                break

def eval_ppl(model, domain, device, bsz=8):
    model.eval()
    ds = DictDataset(domain)
    dl = DataLoader(ds, batch_size=bsz, shuffle=False)
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dl:
            batch = to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            loss = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            tokens = shift_labels.numel()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    avg_loss = total_loss / max(1, total_tokens)
    return perp_from_loss(avg_loss)

# -------------------------
# Methods: adapters-only, EWC, Samsara
# -------------------------
def adapters_only(base_model, tokenizer, domains, device, cfg):
    model = attach_lora(base_model, r=cfg["soul_rank"]).to(device)
    run_train_steps(model, tokenizer, domains["A"], device, steps=cfg["steps"], bsz=cfg["bsz"], lr=cfg["lr"], warmup=cfg["warmup"], trigger_set=None, karmic_w=0.0)
    pA_A = eval_ppl(model, domains["A"], device, bsz=cfg["bsz"])
    run_train_steps(model, tokenizer, domains["B"], device, steps=cfg["steps"], bsz=cfg["bsz"], lr=cfg["lr"], warmup=cfg["warmup"], trigger_set=None, karmic_w=0.0)
    pA_B = eval_ppl(model, domains["A"], device, bsz=cfg["bsz"])
    pB_B = eval_ppl(model, domains["B"], device, bsz=cfg["bsz"])
    return {"ppl_A_after_A": pA_A, "ppl_A_after_B": pA_B, "ppl_B_after_B": pB_B}

def ewc_method(base_model, tokenizer, domains, device, cfg, ewc_lambda=0.05):
    model = base_model.to(device)
    run_train_steps(model, tokenizer, domains["A"], device, steps=cfg["steps"], bsz=cfg["bsz"], lr=cfg["lr"], warmup=cfg["warmup"], trigger_set=None, karmic_w=0.0)
    pA_A = eval_ppl(model, domains["A"], device, bsz=cfg["bsz"])
    fisher = compute_diagonal_fisher(model, tokenizer, domains["A"], device, max_samples=32, bsz=cfg["bsz"])
    theta_star = {n: p.detach().clone() for n, p in model.named_parameters()}
    model.train()
    ds = DictDataset(domains["B"])
    dl = DataLoader(ds, batch_size=cfg["bsz"], shuffle=True)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["lr"])
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=cfg["warmup"], num_training_steps=cfg["steps"])
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    step = 0
    while step < cfg["steps"]:
        for batch in dl:
            batch = to_device(batch, device)
            outputs = model(**batch)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = batch["labels"][:, 1:].contiguous()
            loss_main = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            penalty = torch.tensor(0.0, device=device)
            for n, p in model.named_parameters():
                if n in fisher:
                    penalty = penalty + (fisher[n] * (p - theta_star[n]).pow(2)).sum()
            loss = loss_main + ewc_lambda * penalty
            loss.backward()
            opt.step()
            scheduler.step()
            opt.zero_grad()
            step += 1
            if step >= cfg["steps"]:
                break
    pA_B = eval_ppl(model, domains["A"], device, bsz=cfg["bsz"])
    pB_B = eval_ppl(model, domains["B"], device, bsz=cfg["bsz"])
    return {"ppl_A_after_A": pA_A, "ppl_A_after_B": pA_B, "ppl_B_after_B": pB_B}

def samsara_method(base_model, tokenizer, domains, device, cfg, cremation_strategy="fisher"):
    model = attach_lora(base_model, r=cfg["soul_rank"]).to(device)
    trigger_set = build_trigger_tokens(tokenizer)
    run_train_steps(model, tokenizer, domains["A"], device, steps=cfg["steps"], bsz=cfg["bsz"], lr=cfg["lr"], warmup=cfg["warmup"], trigger_set=trigger_set, karmic_w=cfg["karmic_w"])
    pA_A = eval_ppl(model, domains["A"], device, bsz=cfg["bsz"])
    soul = extract_soul(model)
    base_module = model.base_model if hasattr(model, "base_model") else model
    if cremation_strategy == "random":
        names_to_reset = select_cremation_names_random(base_module, pct=cfg["cremation_pct"], seed=cfg["seed"])
    else:
        fisher = compute_diagonal_fisher(model, tokenizer, domains["A"], device, max_samples=32, bsz=cfg["bsz"])
        names_to_reset = select_cremation_names_fisher(fisher, pct=cfg["cremation_pct"])
    reinit_params_by_name(base_module, names_to_reset)
    load_soul(model, soul)
    run_train_steps(model, tokenizer, domains["A"], device, steps=max(1, cfg["steps"]//4), bsz=cfg["bsz"], lr=cfg["lr"]/2, warmup=max(1, cfg["warmup"]//2), trigger_set=trigger_set, karmic_w=cfg["karmic_w"])
    run_train_steps(model, tokenizer, domains["B"], device, steps=cfg["steps"], bsz=cfg["bsz"], lr=cfg["lr"], warmup=cfg["warmup"], trigger_set=trigger_set, karmic_w=cfg["karmic_w"])
    pA_B = eval_ppl(model, domains["A"], device, bsz=cfg["bsz"])
    pB_B = eval_ppl(model, domains["B"], device, bsz=cfg["bsz"])
    return {"ppl_A_after_A": pA_A, "ppl_A_after_B": pA_B, "ppl_B_after_B": pB_B, "soul": soul, "cremated": list(names_to_reset)}

# -------------------------
# Stats / reporting
# -------------------------
def summarize(all_runs):
    summary = {}
    for method, runs in all_runs.items():
        pA_A = [r["ppl_A_after_A"] for r in runs]
        pA_B = [r["ppl_A_after_B"] for r in runs]
        pB_B = [r["ppl_B_after_B"] for r in runs]
        forgetting = [b - a for a,b in zip(pA_A, pA_B)]
        summary[method] = {
            "ppl_A_after_A_mean": statistics.mean(pA_A),
            "ppl_A_after_A_std": statistics.stdev(pA_A) if len(pA_A)>1 else 0.0,
            "ppl_A_after_B_mean": statistics.mean(pA_B),
            "ppl_A_after_B_std": statistics.stdev(pA_B) if len(pA_B)>1 else 0.0,
            "ppl_B_after_B_mean": statistics.mean(pB_B),
            "ppl_B_after_B_std": statistics.stdev(pB_B) if len(pB_B)>1 else 0.0,
            "forgetting_mean": statistics.mean(forgetting),
            "forgetting_std": statistics.stdev(forgetting) if len(forgetting)>1 else 0.0,
            "raw": runs
        }
    return summary

def paired_ttest(list1, list2):
    if not HAS_SCIPY:
        return None
    res = stats.ttest_rel(list1, list2)
    return {"t_stat": float(res.statistic), "pvalue": float(res.pvalue)}

def transplant_soul_into(model_target, soul, verbose=True):
    mapped = 0
    missing = []
    target_params = dict(model_target.named_parameters())
    with torch.no_grad():
        for name, tensor in soul.items():
            if name in target_params:
                param = target_params[name]
                if param.shape == tensor.shape:
                    param.copy_(tensor.to(param.device))
                    mapped += 1
                else:
                    missing.append(name)
            else:
                missing.append(name)
    if verbose:
        print(f"Transplant: mapped {mapped}/{len(soul)} LoRA params. {len(missing)} missing or mismatch.")
    return {"mapped": mapped, "missing": missing}

# -------------------------
# Main runner
# -------------------------
def run_experiment(args):
    if args.plot and not HAS_MPL:
        print("matplotlib not available; plots disabled.")
        args.plot = False
    if args.trials < 1:
        args.trials = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    domains = build_two_domains(tokenizer, block_size=args.block_size, limit_examples=args.limit_examples)
    all_runs = {"samsara": [], "adapters_only": [], "ewc": []}
    run_meta = []
    for t in range(args.trials):
        seed = args.seed + t*101
        print(f"\n--- Trial {t+1}/{args.trials}, seed={seed} ---")
        set_seed(seed)

        baseA = AutoModelForCausalLM.from_pretrained(args.model)
        baseB = AutoModelForCausalLM.from_pretrained(args.model)
        baseC = AutoModelForCausalLM.from_pretrained(args.model)

        baseA.to(device); baseB.to(device); baseC.to(device)

        cfg = {"steps": args.steps, "bsz": args.bsz, "lr": args.lr, "warmup": args.warmup,
               "soul_rank": args.soul_rank, "cremation_pct": args.cremation_pct, "karmic_w": args.karmic_w, "seed": seed}

        start = time.time()
        res_samsara = samsara_method(baseA, tokenizer, domains, device, cfg, cremation_strategy=args.cremation_strategy)
        elapsed = time.time() - start
        print(f"Samsara done (time {elapsed:.1f}s) PPLs: A->A {res_samsara['ppl_A_after_A']:.2f}, A->B {res_samsara['ppl_A_after_B']:.2f}, B->B {res_samsara['ppl_B_after_B']:.2f}")
        all_runs["samsara"].append(res_samsara)

        start = time.time()
        res_adapt = adapters_only(baseB, tokenizer, domains, device, cfg)
        elapsed = time.time() - start
        print(f"Adapters-only done (time {elapsed:.1f}s) PPLs: A->A {res_adapt['ppl_A_after_A']:.2f}, A->B {res_adapt['ppl_A_after_B']:.2f}, B->B {res_adapt['ppl_B_after_B']:.2f}")
        all_runs["adapters_only"].append(res_adapt)

        start = time.time()
        res_ewc = ewc_method(baseC, tokenizer, domains, device, cfg, ewc_lambda=args.ewc_lambda)
        elapsed = time.time() - start
        print(f"EWC done (time {elapsed:.1f}s) PPLs: A->A {res_ewc['ppl_A_after_A']:.2f}, A->B {res_ewc['ppl_A_after_B']:.2f}, B->B {res_ewc['ppl_B_after_B']:.2f}")
        all_runs["ewc"].append(res_ewc)

        if args.do_transplant:
            print("Attempting cross-architecture transplant of Soul into target model:", args.transplant_target_model)
            target = AutoModelForCausalLM.from_pretrained(args.transplant_target_model).to(device)
            transplant_report = transplant_soul_into(target, res_samsara["soul"], verbose=True)
            try:
                ppl_transplanted = eval_ppl(target, domains["A"], device, bsz=cfg["bsz"])
            except Exception as e:
                print("Transplant evaluation failed:", e)
                ppl_transplanted = None
            run_meta.append({"trial": t, "seed": seed, "transplant_report": transplant_report, "ppl_transplanted": ppl_transplanted})

        run_meta.append({"trial": t, "seed": seed, "cfg": cfg})
        with open(args.out_file, "a") as f:
            f.write(json.dumps({"time": time.time(), "trial": t, "seed": seed, "samsara": {"pplA_A": res_samsara["ppl_A_after_A"], "pplA_B": res_samsara["ppl_A_after_B"]}, "adapters": res_adapt, "ewc": res_ewc}) + "\n")

    summary = summarize(all_runs)

    print("\n======= SUMMARY =======")
    for method, stats in summary.items():
        print(f"\nMethod: {method}")
        print(f"  ppl_A_after_A: {stats['ppl_A_after_A_mean']:.2f} ± {stats['ppl_A_after_A_std']:.2f}")
        print(f"  ppl_A_after_B: {stats['ppl_A_after_B_mean']:.2f} ± {stats['ppl_A_after_B_std']:.2f}")
        print(f"  ppl_B_after_B: {stats['ppl_B_after_B_mean']:.2f} ± {stats['ppl_B_after_B_std']:.2f}")
        print(f"  Forgetting: {stats['forgetting_mean']:.2f} ± {stats['forgetting_std']:.2f}")

    if HAS_SCIPY:
        print("\nPaired t-tests (on forgetting):")
        sams_f = [r["ppl_A_after_B"] - r["ppl_A_after_A"] for r in all_runs["samsara"]]
        adapt_f = [r["ppl_A_after_B"] - r["ppl_A_after_A"] for r in all_runs["adapters_only"]]
        ewc_f = [r["ppl_A_after_B"] - r["ppl_A_after_A"] for r in all_runs["ewc"]]
        t_adapt = paired_ttest(sams_f, adapt_f)
        t_ewc = paired_ttest(sams_f, ewc_f)
        print("Samsara vs Adapters forgetting t-stat:", t_adapt)
        print("Samsara vs EWC forgetting t-stat:", t_ewc)
    else:
        print("\n(scientific t-tests require scipy; install scipy to run statistical tests)")

    out = {"summary": summary, "runs": all_runs, "meta": run_meta}
    with open(args.summary_file, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved summary to", args.summary_file)

    if args.plot and HAS_MPL:
        methods = list(all_runs.keys())
        forgetting_means = [summary[m]["forgetting_mean"] for m in methods]
        forgetting_stds = [summary[m]["forgetting_std"] for m in methods]
        plt.figure(figsize=(6,4))
        plt.bar(methods, forgetting_means, yerr=forgetting_stds, capsize=6)
        plt.title("Forgetting by method (lower better)")
        plt.ylabel("Forgetting (PPL_A_after_B - PPL_A_after_A)")
        plt.savefig("forgetting_by_method.png")
        print("Saved forgetting_by_method.png")

        plt.figure(figsize=(8,4))
        x = range(len(methods))
        a_after_b = [summary[m]["ppl_A_after_B_mean"] for m in methods]
        b_after_b = [summary[m]["ppl_B_after_B_mean"] for m in methods]
        width = 0.35
        plt.bar([i-width/2 for i in x], a_after_b, width=width, label="PPL A after B")
        plt.bar([i+width/2 for i in x], b_after_b, width=width, label="PPL B after B")
        plt.xticks(x, methods)
        plt.legend()
        plt.savefig("ppl_comparison.png")
        print("Saved ppl_comparison.png")

    print("Finished.")

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--soul_rank", type=int, default=8)
    parser.add_argument("--cremation_pct", type=float, default=0.3)
    parser.add_argument("--cremation_strategy", type=str, default="fisher", choices=["fisher","random"])
    parser.add_argument("--karmic_w", type=float, default=0.01)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--limit_examples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ewc_lambda", type=float, default=0.05)
    parser.add_argument("--out_file", type=str, default="runs_log.jsonl")
    parser.add_argument("--summary_file", type=str, default="summary_results.json")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--do_transplant", action="store_true", help="attempt transplant of soul into another model")
    parser.add_argument("--transplant_target_model", type=str, default="gpt2", help="target model for transplant (if do_transplant)")
    args = parser.parse_args()

    run_experiment(args)
