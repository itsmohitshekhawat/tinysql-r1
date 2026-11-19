# TinySQL-R1

**GRPO from scratch: teaching a 0.5B model text-to-SQL with a reward that is checked by running the query.**

Live demo: https://tinysql.mohitshekhawat.com

A small open model (Qwen2.5-0.5B-Instruct) is trained with reinforcement learning. It is never shown a correct
query. For each question it writes 8 attempts, every attempt is executed on a real SQLite database, and the only
signal it gets is whether the rows were right. The training loop is about 150 lines of PyTorch, written from scratch
(no RL library), and ran on a laptop GPU (Apple M-series, about 75 minutes).

The project compares four things on the same 175 test questions: the untrained model, the GRPO-trained model, the
same model trained with supervised fine-tuning (which does see correct queries), and a frontier model over an API.

## Results

Execution accuracy: the model's query returns the right rows, with every column the question asks for.
77 test questions on the 6 training databases (new questions), 98 on 2 databases never seen in training.

| Model | All | Seen DBs | Unseen DBs | Easy | Medium | Hard | Query runs |
|--|--|--|--|--|--|--|--|
| Qwen2.5-0.5B, untrained | 33.7% | 37.7% | 30.6% | 36% | 41% | 24% | 66% |
| + GRPO (reward only) | **49.7%** | 49.4% | **50.0%** | 86% | 37% | 16% | 82% |
| + supervised (sees answers) | 73.7% | 83.1% | 66.3% | 90% | 69% | 58% | 90% |
| Frontier model (DeepSeek V4.1 Flash) | 92.6% | 96.1% | 89.8% | 93% | 90% | 95% | 100% |

What the numbers say:

- **GRPO works, with a ceiling.** +16 points overall, and +19 on databases the model never saw, from a reward alone.
- **The gain is on easy questions.** Hard questions got worse (24% to 16%). The model also learned a lazy habit:
  41% of its answers use `SELECT *` (untrained: 0%). Extra columns are allowed by the check, so this is not wrong,
  but it is a style a reviewer would reject. A reward only shapes what it measures.
- **When you have correct examples, use them.** Supervised training on the same 455 verified questions reaches 74%.
  GRPO is the tool for tasks where you can *check* an answer but cannot *write* one.
- **Training was unstable.** Development accuracy peaked at step 50 (38% to 48%) and then fell. The saved adapter is
  the best checkpoint on a 48-question development set, chosen before the test sets were run.

## Reward hacking: the first run cheated

The first training run reported 34% to 56%. Reading the answers showed why. For

> For each city, show how many customers signed up after 2020 and how many orders those customers placed.

the trained model wrote `SELECT city FROM customers WHERE signup_year > 2020 GROUP BY city`.

The first version of the check accepted any subset of the reference query's columns (to be fair to answers that
leave out helper columns). One easy column with the right number of rows got full reward, and GRPO found that hole
within about 50 steps. Scored fairly, the same model was at 33%: no gain at all.

Fix: every question now lists the reference columns it really asks for (`need` in `data/questions.json`; proposed by
the teacher model, validated by code, see `scripts/label_columns.py`). An answer must contain all of them; extra
columns are allowed. The first run is kept in `data/results/v1-lenient/` and shown on the demo page.

## How GRPO works here

1. **Sample a group.** 4 questions per step, 8 sampled answers each (temperature 1.0).
2. **Check, don't judge.** Run each query read-only. Right rows with the needed columns: 1.0. Runs but wrong: 0.2.
   Fails or is not a query: 0.0.
3. **Compare inside the group.** advantage = (reward - group mean) / group std. No critic network.
4. **Nudge the model.** loss = -mean(advantage x log-probability of the answer tokens). Only a LoRA adapter
   (rank 16, attention + MLP, 8.8M parameters, 1.8% of the model) is trained.

Simplifications, on purpose: one gradient step per batch (so PPO's ratio is 1 and clipping never activates), and no
KL penalty to the starting model. Both are discussed in `train/grpo.py`.

Memory: the vocabulary has 151,000 tokens. Scores for every position of every sequence did not fit in 16 GB, so the
loss runs the transformer body once and computes vocabulary scores only at the answer positions.

## Data

- 8 synthetic SQLite databases (shop, school, library, hospital, flights, hr; held out: movies, sports),
  built by `scripts/make_dbs.py` with a fixed seed.
- 630 question + SQL pairs written by a teacher model and **verified by execution** (`scripts/make_questions.py`):
  455 train (48 of them used as a development set), 77 test on seen databases, 98 test on unseen databases.

## Run it

```bash
# Demo page only (no training libraries needed)
pip install -r requirements.txt
cp .env.example .env            # add an API key for the frontier model
uvicorn app.main:app --port 8000

# Training and evaluation
pip install -r requirements-train.txt
python train/grpo.py --steps 300 --out runs/grpo
python train/sft.py --out runs/sft
python scripts/evaluate.py grpo --adapter runs/grpo
python scripts/evaluate.py deepseek --api
python -m pytest -q tests
```

Live small model (optional): convert the adapter with llama.cpp's `convert_lora_to_gguf.py`, then
`llama-server -m qwen2.5-0.5b-instruct-q4_k_m.gguf --lora tinysql-grpo-lora.gguf --lora-init-without-apply --port 8081`
and set `SMALL_URL`. The page switches the adapter on or off per request, so "before" and "after" share one model
in memory (about 420 MB). Without it, the page shows the small model's recorded answers to the test questions.

## Files

| Path | What |
|--|--|
| `tinysql/sqlenv.py` | prompt, safe read-only execution, the match check and the reward |
| `tinysql/model.py` | load the model (+ adapter), generate |
| `train/grpo.py` | GRPO training loop |
| `train/sft.py` | supervised baseline |
| `scripts/` | build databases, questions, column labels; evaluate |
| `app/` | the demo (FastAPI + plain JS) |
| `data/results/` | every answer of every model, and the training log |

## Limits

The questions were written by a teacher model, so they share its style. The test sets are small. Every score comes
from one training run with one seed. The databases are tiny and synthetic.

MIT licence. Base model: Qwen2.5-0.5B-Instruct (Apache-2.0).
