# Word Embeddings from Scratch (CBOW)

**End-to-end NLP project:** train Continuous Bag-of-Words (CBOW) word embeddings on English news text, inspect semantic quality, and measure transfer to a real classification task.

Built as a Language Modelling coursework project by **Azaliia Agisheva**.

---

- Implementing a classic neural language model (CBOW) in **TensorFlow / Keras**
- Building a full experiment loop: data → training → evaluation → comparison
- Measuring embedding quality both **qualitatively** (nearest neighbors, t-SNE) and **quantitatively** (downstream classification)
- Comparing custom embeddings to a strong public baseline (**GloVe**)
- Tracking hyperparameters and results in reproducible artifact tables

**Stack:** Python, TensorFlow/Keras, NumPy, Pandas, scikit-learn, Matplotlib/Seaborn, NLTK, Gensim, Jupyter

---

## What the project does

1. **Preprocesses** English news corpora (30k and 100k sentences)
2. **Trains CBOW** models with different settings (corpus size, window, embedding dimension, epochs)
3. **Evaluates similarity** for a fixed list of target words (cosine nearest neighbors)
4. **Visualizes** embedding space with t-SNE
5. **Transfers embeddings** into a Reuters newswire topic classifier (frozen vs fine-tuned)
6. **Benchmarks** against random initialization and public GloVe vectors

All code and analysis live in [`word_embeddings.ipynb`](word_embeddings.ipynb). Experiment outputs are saved under [`artifacts/`](artifacts/).

---

## Key results

### CBOW training (validation)

| Setting | Val top-1 | Val top-5 |
|---|---:|---:|
| 100k corpus, window=2, dim=300 | **18.7%** | **33.9%** |
| 100k corpus, window=2, dim=100 | 18.4% | 33.6% |
| 30k corpus, window=2, dim=300 | 16.3% | 30.4% |

Larger corpus and a smaller context window generally improved next-word prediction accuracy in this setup.

### Downstream task: Reuters classification

| Embedding setup | Test accuracy | Macro F1 |
|---|---:|---:|
| Random baseline | 67.5% | 0.11 |
| Best custom CBOW (fine-tuned) | **75.0%** | **0.22** |
| Public GloVe (fine-tuned) | 75.3% | 0.19 |

Custom CBOW embeddings clearly beat random initialization and approach public GloVe performance after fine-tuning — a strong signal that the learned representations capture useful semantic structure.

### Qualitative check

Nearest neighbors look semantically coherent (example from the best 100k / dim=300 run):

- `everyone` → *everybody, anyone, someone, anybody*
- Domain words (e.g. days, roles, institutions) also cluster with related terms

---

## Skills demonstrated

| Area | What was practiced |
|---|---|
| Deep learning | Keras models, embeddings, training loops, callbacks (early stopping, LR schedule) |
| NLP | Tokenization, vocabulary building, CBOW context windows, cosine similarity |
| ML evaluation | Accuracy, top-k, macro F1, frozen vs fine-tuned transfer learning |
| Experimentation | Ablations over corpus size, window size, embedding dim, epochs |
| Engineering | Reproducible seeds, GPU setup, artifact logging (CSV/JSON), notebook pipeline |
| Analysis | Interpreting neighbors, coverage analysis, baseline comparison |

---

## Project structure

```text
├── word_embeddings.ipynb     # Full pipeline (train + evaluate + analyze)
├── target_words.txt          # Words used for similarity inspection
├── eng_news_2024/            # News sentence corpora (30k / 100k)
├── artifacts/
│   ├── embeddings/           # Saved word indices for trained runs
│   ├── tables/               # Metrics, cosine neighbors, Reuters results
│   └── tokenizers/           # Vocabulary files
├── environment.yml           # Conda environment
└── requirements.txt          # Pip dependency freeze
```

---

## How to run

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate LM
jupyter notebook word_embeddings.ipynb
```

### Option B — Pip

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install tensorflow keras numpy pandas scikit-learn matplotlib seaborn nltk gensim jupyter
jupyter notebook word_embeddings.ipynb
```

**Notes**

- GPU is optional but speeds up training (the notebook includes WSL2 / NVIDIA library path helpers).
- Precomputed results are already in `artifacts/tables/` if you only want to inspect outcomes without retraining.
- Set `debug: true` in the notebook config for a fast smoke run on a smaller sentence subset.

---

## Method snapshot

- **Model:** CBOW (predict target word from surrounding context)
- **Data:** English news sentences (`eng_news_2024`)
- **Hyperparameters explored:** window ∈ {2, 5, 8}, dim ∈ {100, 300}, corpus ∈ {30k, 100k}
- **Intrinsic eval:** cosine similarity neighbors + t-SNE
- **Extrinsic eval:** Reuters multi-class topic classification
  - frozen embeddings (representation quality)
  - fine-tuned embeddings (adaptability)
- **Baseline:** GloVe Wiki-Gigaword 100d

---

## Takeaways

1. Word embeddings trained from scratch can transfer to a real NLP classification task.
2. Data scale and context window matter: more data + tighter windows helped CBOW accuracy here.
3. Fine-tuning pretrained vectors consistently outperformed freezing them.
4. Public GloVe still wins slightly on accuracy, but custom CBOW is competitive and stronger on macro F1 in the best run — useful for discussing trade-offs in interviews.

---


*Language Modelling course project (2025–2026).*
