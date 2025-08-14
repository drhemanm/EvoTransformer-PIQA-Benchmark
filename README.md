
# EvoTransformer — PIQA Benchmark (Reproducible)

**Claim:** A small-footprint EvoTransformer model outperforms GPT-3.5 on the PIQA commonsense reasoning benchmark (validation split), under identical conditions.

---

## 📊 TL;DR (Results)

- **Dataset:** PIQA validation (N = 1838, from TFDS)
- **EvoTransformer:** 59.41% accuracy
- **GPT-3.5 (0-shot, A/B choice):** 50.54% accuracy
- **McNemar exact test:** p ≈ 4.086 × 10⁻⁸ (statistically significant)
- **95% bootstrap CI (Evo – GPT):** [0.0555, 0.1219]

_All code, predictions, and metrics are included for full auditability and reproducibility._

---

## 📦 Environment Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**On Google Colab:**
```bash
!pip install -q -r requirements.txt
```

---

## 🚀 1) Train EvoTransformer on PIQA

Trains on the **full PIQA/train** split and early-stops on the **validation** split.

```bash
python train_piqa.py
```

**Artifacts produced:**
- `results/evo_piqa_best.pt` — Best model weights
- `results/train_summary.json` — Seed, best accuracy, timestamp

---

## 🆚 2) Evaluate Evo vs GPT-3.5 (Full Validation)

**Set your OpenAI API key** before running GPT-3.5 evaluation:

```bash
export OPENAI_API_KEY=sk-xxxx      # macOS/Linux
# setx OPENAI_API_KEY sk-xxxx      # Windows PowerShell
```

Then run:

```bash
python eval_piqa.py
```

**Artifacts produced:**
- `results/preds_val_evo_full.csv` — Evo predictions
- `results/preds_val_gpt_full.csv` — GPT-3.5 predictions
- `results/metrics.json` — Final accuracy, McNemar p-value, bootstrap CI

**Sample `results/metrics.json`:**
```json
{
  "N": 1838,
  "evo_acc": 0.5941,
  "gpt_acc": 0.5054,
  "b_gpt_only_correct": 377,
  "c_evo_only_correct": 540,
  "mcnemar_exact_p": 4.086e-08,
  "bootstrap_gap_95CI": [0.0555, 0.1219]
}
```

---

## 🔄 Reproducibility

- Fixed seeds across `random`, `numpy`, `torch`, and `torch.cuda`.
- Deterministic PyTorch ops (including attention path) enabled.
- Exact library versions pinned in `requirements.txt`.
- Full predictions saved to CSV so results can be recomputed without re-running models.

---

## 📌 Notes

- PIQA **test** labels are not public; validation is the standard public benchmark split.
- GPT-3.5 is run 0-shot with `temperature=0` for deterministic outputs.
- EvoTransformer here is a compact encoder model trained specifically for PIQA; architecture and training loop are included in this repo.

---

## 📚 Citation

If you use this repo, please cite:

```
Mohabeer, H. (2025). EvoTransformer — PIQA Benchmark (Reproducible).
```

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
