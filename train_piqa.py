# train_piqa.py
import os, json, time, numpy as np, torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from utils_common import set_repro, PIQAPairs, EvoPIQAModel, tokenize_pair

def collate_pairs(batch_rows, max_length, device):
    # Flatten pairs → pad once → unflatten to [B,2]
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    flat_ids, flat_mask, labels = [], [], []
    for r in batch_rows:
        goal = r["goal"].decode(); sol1 = r["sol1"].decode(); sol2 = r["sol2"].decode()
        labels.append(int(r["label"]))
        enc = tok([goal+" "+sol1, goal+" "+sol2], truncation=True, max_length=max_length, return_tensors="pt", padding=True)
        flat_ids.append(enc["input_ids"]); flat_mask.append(enc["attention_mask"])
    flat_ids = torch.cat(flat_ids, dim=0).to(device)
    flat_mask = torch.cat(flat_mask, dim=0).to(device)
    labels = torch.tensor(labels).to(device)
    return {"input_ids": flat_ids, "attention_mask": flat_mask}, labels

def train_eval(seed=42, epochs=8, batch_size=48, max_length=64, lr=3e-4, weight_decay=0.01, patience=3):
    set_repro(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = PIQAPairs("train")
    val_ds   = PIQAPairs("validation")
    train_loader = DataLoader(train_ds.rows, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_pairs(x, max_length, device))
    val_loader   = DataLoader(val_ds.rows,   batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_pairs(x, max_length, device))

    model = EvoPIQAModel().to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    scaler = GradScaler('cuda') if torch.cuda.is_available() else None

    def pairwise_loss(scores, labels):
        # scores: [B,2] (A,B). labels: 0 if B is correct, 1 if A is correct
        target = labels.float()
        diff = scores[:,0] - scores[:,1]
        return nn.BCEWithLogitsLoss()(diff, target)

    def evaluate():
        model.eval(); correct=total=0
        with torch.no_grad(), autocast('cuda', enabled=torch.cuda.is_available()):
            for pack, labels in val_loader:
                logits = model(pack["input_ids"], pack["attention_mask"]).view(-1, 2)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item(); total += labels.size(0)
        return correct / total

    best_acc, bad, best_state = 0.0, 0, None
    for ep in range(1, epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}")
        for pack, labels in pbar:
            with autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(pack["input_ids"], pack["attention_mask"]).view(-1,2)
                loss = pairwise_loss(logits, labels)
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            pbar.set_postfix(loss=float(loss))
        sched.step()
        acc = evaluate()
        print(f"✅ Val acc (full split): {acc:.4f}")
        if acc > best_acc:
            best_acc, bad = acc, 0
            best_state = {k: v.cpu() for k,v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("⏹️ Early stopping"); break

    os.makedirs("results", exist_ok=True)
    torch.save(best_state, f"results/evo_piqa_best.pt")
    with open("results/train_summary.json","w") as f:
        json.dump({"seed": seed, "best_val_acc": best_acc, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    return best_acc

if __name__ == "__main__":
    # example multi-seed
    seeds = [13,21,42]
    accs = [train_eval(seed=s, epochs=8, batch_size=48, max_length=64) for s in seeds]
    print(f"Mean±Std over {len(seeds)} seeds: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
