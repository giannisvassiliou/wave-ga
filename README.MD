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

## Repository contents

The repository includes the paper PDF and all experiment drivers used in the empirical evaluation.

### Core files

- `GECCO_2026_Wave_GA_FINAL.pdf`  
  Full paper describing the Wave-GA method, motivation, and experimental results.

- `synthetic_exps.py`  
  Synthetic streaming experiments and ablations, including severe drift, recurring drift, sudden drift, cycling classes, and extreme class imbalance. Includes Wave-GA, ablations, and evolutionary baselines with statistical testing.

- `airlines.py`  
  Real-world streaming experiment on the Airlines dataset.

- `elect2.py`  
  Real-world streaming experiment on the Electricity Market (Elec2) dataset.

- `weather.py`  
  Real-world streaming experiment on a Weather dataset (e.g., `weatherAUS.csv`).

---

## Method overview

Wave-GA maintains and evolves a **linear multiclass classifier** with weight matrix:

W ∈ ℝ^(C × d)

Predictions are made via:

ŷ = argmax_c (w_cᵀ x)

### Key components

1. **Class-focused training waves**  
   Training proceeds in cyclical waves, each focusing on a specific class to prevent majority domination.

2. **Blended minority-aware fitness**  

   F(h) = α · f_class(h, c*) + (1 − α) · f_overall(h)

3. **Per-class replay buffers**  
   Buffers mitigate catastrophic forgetting and improve drift adaptation.

4. **Dynamic class expansion**  
   New classes trigger expansion of the weight matrix while preserving learned parameters.

---

## Installation

### Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Install dependencies

```bash
pip install numpy scipy pandas matplotlib scikit-learn river scikit-posthocs
```

---

## Running experiments

### Synthetic experiments

```bash
python synthetic_exps.py
```

### Electricity Market

```bash
python elect2.py
```

### Airlines

```bash
python airlines.py
```

### Weather

```bash
python weather.py
```

---

## Evaluation

Experiments follow a **prequential (test-then-train)** protocol and report metrics such as Balanced Accuracy and F1. Results are averaged across multiple runs with statistical tests.

---

## Reproducibility

- Fixed random seeds per run
- Deterministic data ordering when applicable

---

## Citation

Wave-GA: Cyclical Evolutionary Optimization for Streaming Classification with Emerging Labels  
Anonymous Author(s), GECCO 2026 (submission)

---

## License

Add your preferred open-source license.
