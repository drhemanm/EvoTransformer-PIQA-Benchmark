# utils_common.py
import os, random, numpy as np, torch, csv
import tensorflow_datasets as tfds
from transformers import AutoTokenizer
import torch.nn as nn
from math import comb

# ---------------- Reproducibility ----------------
def set_repro(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # force deterministic math path for attention
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

# ---------------- Dataset ----------------
class PIQAPairs:
    """Lightweight wrapper over TFDS 'piqa' split; rows are raw items for flexible tokenization."""
    def __init__(self, split="train"):
        self.ds = tfds.load("piqa", split=split, as_supervised=False)
        self.rows = list(tfds.as_numpy(self.ds))

# ---------------- Model ----------------
class EvoPIQAModel(nn.Module):
    """Small-footprint transformer encoder with masked-mean pooling + pairwise scoring."""
    def __init__(self, d_model=384, nhead=6, num_layers=6, dim_feedforward=1536, vocab_size=30522):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)  # one score per sequence

    def masked_mean(self, x, mask):
        lengths = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B,1]
        return (x * mask.unsqueeze(-1)).sum(dim=1) / lengths

    def forward(self, input_ids, attention_mask):
        x = self.embed(input_ids)  # [B,T,D]
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        pooled = self.masked_mean(x, attention_mask)
        return self.head(pooled).squeeze(-1)  # [B] score

# ---------------- Tokenization helpers ----------------
_tokenizer = None
def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return _tokenizer

def tokenize_pair(goal, sol1, sol2, max_length=64, device="cpu"):
    tok = get_tokenizer()
    enc = tok([goal+" "+sol1, goal+" "+sol2],
              truncation=True, max_length=max_length, return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items()}  # dict with input_ids, attention_mask

# ---------------- Stats ----------------
def mcnemar_from_csv(evo_csv, gpt_csv):
    e = {int(r["idx"]):(int(r["label"]), int(r["evo_pred"])) for r in csv.DictReader(open(evo_csv))}
    g = {int(r["idx"]):int(r["gpt_pred"]) for r in csv.DictReader(open(gpt_csv))}
    ids = sorted(set(e.keys()) & set(g.keys()))
    evo_acc = sum(1 for i in ids if e[i][1]==e[i][0]) / len(ids)
    gpt_acc = sum(1 for i in ids if g[i]   ==e[i][0]) / len(ids)
    b = sum(1 for i in ids if g[i]==e[i][0] and e[i][1]!=e[i][0])
    c = sum(1 for i in ids if e[i][1]==e[i][0] and g[i]!=e[i][0])
    n=b+c; k=min(b,c)
    p = sum(comb(n,j) for j in range(0,k+1)) * (0.5**n) if n>0 else 1.0
    return evo_acc, gpt_acc, b, c, p, len(ids)

def bootstrap_gap_ci(evo_csv, gpt_csv, B=2000, seed=42):
    import numpy as np, random
    random.seed(seed); np.random.seed(seed)
    e = {int(r["idx"]):(int(r["label"]), int(r["evo_pred"])) for r in csv.DictReader(open(evo_csv))}
    g = {int(r["idx"]):int(r["gpt_pred"]) for r in csv.DictReader(open(gpt_csv))}
    ids = sorted(set(e.keys()) & set(g.keys()))
    y  = np.array([e[i][0] for i in ids])
    eok= np.array([e[i][1]==y[k] for k,i in enumerate(ids)], dtype=np.int32)
    gok= np.array([g[i]    ==y[k] for k,i in enumerate(ids)], dtype=np.int32)
    gaps=[]; n=len(ids)
    for _ in range(B):
        idx = np.random.randint(0,n,size=n)
        gaps.append(eok[idx].mean()-gok[idx].mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return float(lo), float(hi)
