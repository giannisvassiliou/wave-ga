# Wave-GA: Cyclical Evolutionary Optimization for Streaming Classification with Emerging Labels

This repository contains the **reference implementation and experimental code** for the paper:

**Wave-GA: Cyclical Evolutionary Optimization for Streaming Classification with Emerging Labels**  
(Submitted to GECCO 2026)

Wave-GA is a **population-based evolutionary framework** for **streaming multiclass classification** under the “holy trinity” of non-stationarity:

- **Concept drift**
- **Extreme class imbalance**
- **Evolving / emerging label spaces**

The method evolves a **single linear weight matrix** using **class-focused training waves**, a **minority-aware blended fitness**, **per-class replay buffers**, and **dynamic model expansion** when new classes appear.

---

## Repository structure

The repository follows a simple and reproducible layout:

```
.
├── main/
│   ├── synthetic_exps.py
│   ├── airlines.py
│   ├── elect2.py
│   └── weather.py
│
├── lib&data/
│   ├── ultra_fast_wave_ga_*.py
│   ├── ultra_fast_streaming_de_*.py
│   ├── linear_streaming_classifiers.py
│   └── *
│   ├── weatherAUS.csv       # user-provided
└── README.md
```

- **`main/`** contains all **entry-point experiment scripts**
- **`lib&data/`** contains Wave-GA implementations, baselines, and shared utilities
              ** contains local datasets (optional; some streams auto-download)

All experiments are designed to run **directly  with ALL THE LIBRARIES PRESENT IN THE SAME FOLDER**.

---

## Repository contents

### Core files

- `main/synthetic_exps.py`  
  Synthetic streaming experiments and ablations, including:
  - Severe drift
  - Recurring drift
  - Sudden drift
  - Cycling classes
  - Extreme class imbalance  
  Includes Wave-GA, ablations, and evolutionary baselines with statistical testing.

- `main/airlines.py`  
  Real-world streaming experiment on the **Airlines** dataset.

- `main/elect2.py`  
  Real-world streaming experiment on the **Electricity Market (Elec2)** dataset.

- `main/weather.py`  
  Real-world streaming experiment on a **Weather** dataset (e.g., `weatherAUS.csv`).

---

## Method overview

Wave-GA maintains and evolves a **linear multiclass classifier** with weight matrix:

```
W ∈ ℝ^(C × d)
```

Predictions are made via:

```
ŷ = argmax_c (w_cᵀ x)
```

### Key components

1. **Class-focused training waves**  
   Cyclical optimization phases ensure sustained pressure on minority and emerging classes.

2. **Blended minority-aware fitness**  

   ```
   F(h) = α · f_class(h, c*) + (1 − α) · f_overall(h)
   ```

3. **Per-class replay buffers**  
   Reduce catastrophic forgetting and improve adaptation to drift.

4. **Dynamic class expansion**  
   The model expands automatically when unseen labels appear in the stream.

---


### Install dependencies

```bash
pip install numpy scipy pandas matplotlib scikit-learn river scikit-posthocs numba
Numba is needed to  speedup the calculations
```

---

## Running experiments

### Important notes

- **No command-line flags are required**
- Each experiment script runs **out of the box**
- Scripts automatically:
  - execute multiple runs
  - report aggregate statistics
  - print runtime measurements

### Synthetic experiments

```bash
cd main
python synthetic_exps.py
```

Runs all synthetic benchmarks, ablations, and baselines, and reports:
- mean ± std performance
- statistical significance tests
- total and per-run execution time

---

### Electricity Market (Elec2)

```bash
cd main
python elect2.py
```

- Automatically loads or downloads the dataset
- Reports accuracy metrics, statistical summaries, and runtime

---

### Airlines dataset

```bash
cd main
python airlines.py
```

- Uses a local `airlines.csv` if available, otherwise falls back to automatic loading
- Reports full statistical summaries and runtime

---

### Weather dataset

```bash
cd main
python weather.py
```

- Expects a local CSV file (e.g., `weatherAUS.csv`) in `data/`
- Reports performance metrics, statistics, and runtime

---

## Evaluation protocol

All experiments use a **prequential (test-then-train)** streaming evaluation protocol.

Reported metrics include:
- Balanced Accuracy (BA)
- F1 score
- Minority-F1 / MinF1 (where applicable)

Results are averaged over multiple runs and accompanied by statistical tests.

---

## Reproducibility

- Fixed random seeds per run
- Deterministic data ordering when applicable
- Explicit multi-run evaluation baked into each experiment script

---
