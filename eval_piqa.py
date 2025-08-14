# eval_piqa.py
import os, csv, json, time, random, torch
from tqdm import tqdm
from utils_common import EvoPIQAModel, PIQAPairs, tokenize_pair, mcnemar_from_csv, bootstrap_gap_ci, set_repro

# --------- Config ----------
MAX_LEN = 64
EVO_WEIGHTS = "results/evo_piqa_best.pt"   # from training step
EVO_CSV = "results/preds_val_evo_full.csv"
GPT_CSV = "results/preds_val_gpt_full.csv"
RESULTS_JSON = "results/metrics.json"
# ---------------------------

def dump_evo_preds(weights_path=EVO_WEIGHTS, out_csv=EVO_CSV, max_length=MAX_LEN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvoPIQAModel().to(device)
    sd = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    val = PIQAPairs("validation")
    rows=[]
    with torch.no_grad():
        for idx, r in enumerate(tqdm(val.rows, desc="Evo predicting")):
            goal = r["goal"].decode(); sol1 = r["sol1"].decode(); sol2 = r["sol2"].decode()
            y = int(r["label"])
            enc = tokenize_pair(goal, sol1, sol2, max_length=max_length, device=device)
            logits = model(enc["input_ids"], enc["attention_mask"]).view(-1,2)
            probs = torch.softmax(logits, dim=1)
            pred = int(torch.argmax(probs, dim=1).item())
            rows.append((idx, y, pred, float(probs[0,0]), float(probs[0,1])))
    os.makedirs("results", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","label","evo_pred","scoreA","scoreB"]); w.writerows(rows)
    print(f"wrote {out_csv} with {len(rows)} rows")

def dump_gpt_preds(out_csv=GPT_CSV, model_name="gpt-3.5-turbo", limit=None, sleep_s=0.0):
    """
    Requires OPENAI_API_KEY in env. Uses legacy openai.ChatCompletion for simplicity.
    """
    import openai, tensorflow_datasets as tfds
    ds = tfds.load("piqa", split="validation", as_supervised=False)
    rows=[]
    for idx, ex in tqdm(enumerate(tfds.as_numpy(ds)), desc="GPT predicting"):
        if limit is not None and idx >= limit: break
        goal = ex["goal"].decode(); sol1 = ex["sol1"].decode(); sol2 = ex["sol2"].decode()
        y = int(ex["label"])
        prompt = f"""Choose the better solution to achieve the goal.

Goal: {goal}

Option A: {sol1}
Option B: {sol2}

Answer with "A" or "B" only."""
        try:
            res = openai.ChatCompletion.create(
                model=model_name,
                messages=[{"role":"user","content": prompt}],
                temperature=0,
            )
            ans = res["choices"][0]["message"]["content"].strip().upper()
            pred = 0 if ans.startswith("A") else 1
        except Exception as e:
            pred = random.randint(0,1)
        rows.append((idx, y, pred))
        if sleep_s>0: time.sleep(sleep_s)
    os.makedirs("results", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["idx","label","gpt_pred"]); w.writerows(rows)
    print(f"wrote {out_csv} with {len(rows)} rows")

if __name__ == "__main__":
    set_repro(42)

    # 1) Evo predictions
    if not os.path.exists(EVO_CSV):
        dump_evo_preds()

    # 2) GPT predictions (full set). Make sure OPENAI_API_KEY is set; you can test with limit=100 first.
    if not os.path.exists(GPT_CSV):
        # Example quick test: dump_gpt_preds(out_csv=GPT_CSV, limit=200)
        dump_gpt_preds(out_csv=GPT_CSV, limit=None)

    # 3) Stats
    evo_acc, gpt_acc, b, c, p, N = mcnemar_from_csv(EVO_CSV, GPT_CSV)
    lo, hi = bootstrap_gap_ci(EVO_CSV, GPT_CSV, B=2000)

    metrics = {
        "N": N, "evo_acc": round(evo_acc, 4), "gpt_acc": round(gpt_acc,4),
        "b_gpt_only_correct": b, "c_evo_only_correct": c, "mcnemar_exact_p": p,
        "bootstrap_gap_95CI": [round(lo,4), round(hi,4)]
    }
    print(json.dumps(metrics, indent=2))
    with open(RESULTS_JSON,"w") as f: json.dump(metrics, f, indent=2)
    print(f"Saved {RESULTS_JSON}")
