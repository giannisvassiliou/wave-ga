"""
Ultra-Fast Differential Evolution with Numba JIT Compilation
==============================================================

Numba-accelerated versions of:
1. UltraFastStreamingDE_Waves: DE with wave-based class-focused training
2. UltraFastStandardDE: DE without waves (baseline)

Expected: 20-50x speedup over original implementations

Prerequisites:
    pip install numba
"""

import time
import numpy as np
from numba import jit, prange
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# =====================================================================

@jit(nopython=True, cache=True, fastmath=True)
def numba_predict(X, W):
    """Fast prediction: argmax(X @ W.T)"""
    n_samples = X.shape[0]
    n_classes = W.shape[0]
    
    y_pred = np.zeros(n_samples, dtype=np.int32)
    
    for s in range(n_samples):
        max_score = -1e10
        max_class = 0
        
        for c in range(n_classes):
            score = 0.0
            for f in range(X.shape[1]):
                score += W[c, f] * X[s, f]
            
            if score > max_score:
                max_score = score
                max_class = c
        
        y_pred[s] = max_class
    
    return y_pred


@jit(nopython=True, cache=True, fastmath=True)
def numba_balanced_accuracy(y_true, y_pred, n_classes):
    """Ultra-fast multiclass balanced accuracy"""
    n = len(y_true)
    if n == 0:
        return 0.5
    
    recalls = np.zeros(n_classes, dtype=np.float32)
    
    for c in range(n_classes):
        n_true = 0
        n_correct = 0
        
        for i in range(n):
            if y_true[i] == c:
                n_true += 1
                if y_pred[i] == c:
                    n_correct += 1
        
        if n_true > 0:
            recalls[c] = n_correct / n_true
        else:
            recalls[c] = 0.0
    
    return np.mean(recalls)


@jit(nopython=True, cache=True, fastmath=True)
def numba_binary_balanced_accuracy(y_true, y_pred):
    """Ultra-fast binary balanced accuracy"""
    n = len(y_true)
    if n == 0:
        return 0.5
    
    tp = fp = tn = fn = 0
    for i in range(n):
        if y_true[i] == 1:
            tp += (y_pred[i] == 1)
            fn += (y_pred[i] == 0)
        else:
            fp += (y_pred[i] == 1)
            tn += (y_pred[i] == 0)
    
    pos = tp + fn
    neg = tn + fp
    
    sensitivity = tp / pos if pos > 0 else 0.0
    specificity = tn / neg if neg > 0 else 0.0
    
    return (sensitivity + specificity) / 2.0


@jit(nopython=True, cache=True, fastmath=True)
def numba_fitness_blended(W, X, y, target_class, n_classes, alpha):
    """
    Blended fitness for wave training:
    alpha * BA_class + (1-alpha) * BA_overall
    """
    n_samples = X.shape[0]
    
    # Predictions
    y_pred = numba_predict(X, W)
    
    # Overall BA
    ba_overall = numba_balanced_accuracy(y, y_pred, n_classes)
    
    # Class-focused BA (binary: target vs rest)
    y_binary = np.zeros(n_samples, dtype=np.int32)
    y_pred_binary = np.zeros(n_samples, dtype=np.int32)
    
    for i in range(n_samples):
        y_binary[i] = (y[i] == target_class)
        y_pred_binary[i] = (y_pred[i] == target_class)
    
    ba_class = numba_binary_balanced_accuracy(y_binary, y_pred_binary)
    
    # Blended
    return alpha * ba_class + (1.0 - alpha) * ba_overall


@jit(nopython=True, cache=True, fastmath=True)
def numba_fitness_overall(W, X, y, n_classes):
    """Overall fitness only (no blending)"""
    y_pred = numba_predict(X, W)
    return numba_balanced_accuracy(y, y_pred, n_classes)


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def numba_evaluate_population_blended(population, X, y, target_class, n_classes, alpha):
    """
    Parallel fitness evaluation for entire population (blended fitness)
    
    Args:
        population: (pop_size, n_classes, n_features)
        X: (n_samples, n_features)
        y: (n_samples,)
        target_class: int
        n_classes: int
        alpha: float
    
    Returns:
        fitnesses: (pop_size,)
    """
    pop_size = population.shape[0]
    fitnesses = np.zeros(pop_size, dtype=np.float32)
    
    for i in prange(pop_size):
        fitnesses[i] = numba_fitness_blended(
            population[i], X, y, target_class, n_classes, alpha
        )
    
    return fitnesses


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def numba_evaluate_population_overall(population, X, y, n_classes):
    """
    Parallel fitness evaluation for entire population (overall fitness only)
    
    Args:
        population: (pop_size, n_classes, n_features)
        X: (n_samples, n_features)
        y: (n_samples,)
        n_classes: int
    
    Returns:
        fitnesses: (pop_size,)
    """
    pop_size = population.shape[0]
    fitnesses = np.zeros(pop_size, dtype=np.float32)
    
    for i in prange(pop_size):
        fitnesses[i] = numba_fitness_overall(population[i], X, y, n_classes)
    
    return fitnesses


@jit(nopython=True, cache=True, fastmath=True)
def numba_de_mutation(population, idx, F):
    """DE/rand/1 mutation with clipping"""
    pop_size = population.shape[0]
    
    # Select 3 random individuals (different from idx)
    candidates = np.empty(pop_size - 1, dtype=np.int32)
    j = 0
    for i in range(pop_size):
        if i != idx:
            candidates[j] = i
            j += 1
    
    # Randomly select 3
    selected = np.random.choice(candidates, 3, replace=False)
    r1, r2, r3 = selected[0], selected[1], selected[2]
    
    # Mutation: v = x_r1 + F * (x_r2 - x_r3)
    mutant = population[r1] + F * (population[r2] - population[r3])
    
    # Clip to prevent explosion
    mutant = np.clip(mutant, -10.0, 10.0)
    
    return mutant


@jit(nopython=True, cache=True, fastmath=True)
def numba_de_crossover(target, mutant, CR):
    """DE binomial crossover"""
    n_classes, n_features = target.shape
    trial = target.copy()
    
    j_rand = np.random.randint(0, n_features)
    
    for i in range(n_classes):
        for j in range(n_features):
            if np.random.rand() < CR or j == j_rand:
                trial[i, j] = mutant[i, j]
    
    return trial


# =====================================================================
# ULTRA-FAST STREAMING DE WITH WAVES
# =====================================================================
class UltraFastStreamingDE_Waves:
    """
    Ultra-Fast Differential Evolution with Wave-Based Training
    Behavior-matched to StreamingDE_Waves (but using Numba JIT helpers).
    """
    
    def __init__(
        self,
        n_features,
        population_size=25,
        generations_per_wave=8,
        n_cycles=3,
        F=0.8,
        CR=0.9,
        buffer_size=300,      # match buffer_size_per_class=200 in non-numba
        alpha=0.7,
        seed=None,
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.generations_per_wave = generations_per_wave
        self.n_cycles = n_cycles
        self.F = F
        self.CR = CR
        self.buffer_size = buffer_size
        self.alpha = alpha
        
        if seed is not None:
            np.random.seed(seed)
        
        # Population as numpy array for Numba
        self.population = None  # (pop_size, n_classes, n_features)
        self.best_chromosome = None
        
        # Class management
        self.n_classes = 0
        self.seen_classes = []
        
        # Per-class circular buffers
        self.class_buffers_X = {}
        self.class_buffers_ptr = {}
        self.class_buffers_size = {}
        
        # Stats
        self.training_time = 0.0
        self.n_updates = 0
    
    def _initialize_population(self):
        """Initialize population like StreamingDE_Waves: N(0, 0.1)."""
        self.population = (
            np.random.randn(
                self.population_size, self.n_classes, self.n_features
            ).astype(np.float32)
            * 0.1
        )
    
    def _update_class_buffer(self, c, X_c):
        """Update circular buffer for class c (same semantics as 'sliding buffer')."""
        if len(X_c) == 0:
            return
        
        if c not in self.class_buffers_X:
            self.class_buffers_X[c] = np.zeros(
                (self.buffer_size, self.n_features), dtype=np.float32
            )
            self.class_buffers_ptr[c] = 0
            self.class_buffers_size[c] = 0
        
        buffer = self.class_buffers_X[c]
        ptr = self.class_buffers_ptr[c]
        size = self.class_buffers_size[c]
        n_samples = len(X_c)
        
        if n_samples >= self.buffer_size:
            buffer[:] = X_c[-self.buffer_size:]
            self.class_buffers_size[c] = self.buffer_size
            self.class_buffers_ptr[c] = 0
        else:
            end_ptr = ptr + n_samples
            if end_ptr <= self.buffer_size:
                buffer[ptr:end_ptr] = X_c
            else:
                overflow = end_ptr - self.buffer_size
                buffer[ptr:] = X_c[: self.buffer_size - ptr]
                buffer[:overflow] = X_c[self.buffer_size - ptr :]
            
            self.class_buffers_ptr[c] = end_ptr % self.buffer_size
            self.class_buffers_size[c] = min(size + n_samples, self.buffer_size)
    
    def _sample_from_buffer(self, c, n):
        """Sample n examples from class c buffer (no replacement)."""
        if c not in self.class_buffers_X:
            return np.zeros((0, self.n_features), dtype=np.float32)
        
        size = self.class_buffers_size[c]
        if size == 0:
            return np.zeros((0, self.n_features), dtype=np.float32)
        
        n = min(n, size)
        indices = np.random.choice(size, size=n, replace=False)
        
        buffer = self.class_buffers_X[c]
        if size < self.buffer_size:
            return buffer[indices].copy()
        else:
            ptr = self.class_buffers_ptr[c]
            actual_indices = (ptr + indices) % self.buffer_size
            return buffer[actual_indices].copy()
    
    def _build_wave_batch(self, target_class, wave_size=250):
        """
        Construct balanced mini-batch for target class.
        
        Matches StreamingDE_Waves._construct_wave_batch:
        - wave_size = 250
        - 50% from target class
        - 50% from other classes (proportionally)
        """
        X_list = []
        y_list = []
        
        # ----- target class samples -----
        n_target_actual = 0
        if target_class in self.class_buffers_X:
            size_t = self.class_buffers_size[target_class]
            if size_t > 0:
                n_target = min(wave_size // 2, size_t)
                X_target = self._sample_from_buffer(target_class, n_target)
                n_target_actual = len(X_target)
                if n_target_actual > 0:
                    y_target = np.full(n_target_actual, target_class, dtype=np.int32)
                    X_list.append(X_target)
                    y_list.append(y_target)
        
        # ----- other class samples -----
        other_classes = [c for c in self.class_buffers_X if c != target_class]
        n_other = wave_size - n_target_actual
        
        if other_classes and n_other > 0:
            per_class = max(1, n_other // len(other_classes))
            for cls in other_classes:
                size_c = self.class_buffers_size.get(cls, 0)
                if size_c > 0:
                    # Sample up to per_class from this class
                    X_c = self._sample_from_buffer(cls, per_class)
                    if len(X_c) > 0:
                        y_c = np.full(len(X_c), cls, dtype=np.int32)
                        X_list.append(X_c)
                        y_list.append(y_c)
        
        if not X_list:
            return None, None
        
        X_wave = np.vstack(X_list)
        y_wave = np.concatenate(y_list)
        
        # Shuffle (same idea as non-numba)
        shuffle_idx = np.random.permutation(len(X_wave))
        return X_wave[shuffle_idx], y_wave[shuffle_idx]
    
    def _wave_training(self, X, y, target_class, n_generations):
        """Numba-accelerated DE evolution on a wave batch using blended fitness."""
        for _ in range(n_generations):
            new_population = np.zeros_like(self.population)
            
            for idx in range(self.population_size):
                target = self.population[idx]
                
                # Mutation
                mutant = numba_de_mutation(self.population, idx, self.F)
                
                # Crossover
                trial = numba_de_crossover(target, mutant, self.CR)
                
                # Selection using blended fitness
                f_target = numba_fitness_blended(
                    target, X, y, target_class, self.n_classes, self.alpha
                )
                f_trial = numba_fitness_blended(
                    trial, X, y, target_class, self.n_classes, self.alpha
                )
                
                if f_trial >= f_target:
                    new_population[idx] = trial
                else:
                    new_population[idx] = target
            
            self.population = new_population
    
    def partial_fit(self, X, y):
        """
        Online update with wave-based DE training.
        
        Structure mirrors StreamingDE_Waves.partial_fit:
        1. Detect/expand classes
        2. Update buffers
        3. For each of n_cycles:
           - For each class:
             * Construct wave batch (wave_size=250)
             * Run DE for generations_per_wave
        4. Select best chromosome with overall BA.
        """
        start_time = time.time()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        # ---- detect new classes and expand population ----
        unique_classes = np.unique(y)
        new_classes = set(unique_classes) - set(self.seen_classes)
        
        if new_classes:
            old_n_classes = self.n_classes
            
            for c in sorted(new_classes):
                self.seen_classes.append(int(c))
            self.seen_classes.sort()
            self.n_classes = len(self.seen_classes)
            
            if self.population is not None:
                old_pop = self.population
                new_pop = np.zeros(
                    (self.population_size, self.n_classes, self.n_features),
                    dtype=np.float32,
                )
                # keep old weights
                new_pop[:, :old_n_classes, :] = old_pop
                # random init for new classes (match non-numba expand)
                n_new = self.n_classes - old_n_classes
                if n_new > 0:
                    new_pop[:, old_n_classes:, :] = (
                        np.random.randn(
                            self.population_size, n_new, self.n_features
                        ).astype(np.float32)
                        * 0.1
                    )
                self.population = new_pop
        else:
            if self.n_classes == 0:
                # first batch ever
                self.seen_classes = sorted(int(c) for c in unique_classes)
                self.n_classes = len(self.seen_classes)
        
        # init population if needed
        if self.population is None and self.n_classes > 0:
            self._initialize_population()
        
        # ---- update buffers ----
        for c in unique_classes:
            c = int(c)
            X_c = X[y == c]
            self._update_class_buffer(c, X_c)
        
        # classes that currently have buffers
        active_classes = list(self.class_buffers_X.keys())
        if len(active_classes) == 0:
            return self
        
        # ---- run waves: fixed n_cycles & generations_per_wave (no adaptivity) ----
        for _ in range(self.n_cycles):
            for target_class in active_classes:
                X_wave, y_wave = self._build_wave_batch(target_class, wave_size=250)
                if X_wave is not None and len(X_wave) > 0:
                    self._wave_training(
                        X_wave, y_wave, target_class, self.generations_per_wave
                    )
        
        # ---- select best chromosome based on overall BA (all buffered data) ----
        X_all_list = []
        y_all_list = []
        for c in self.seen_classes:
            if c not in self.class_buffers_X:
                continue
            size = self.class_buffers_size[c]
            if size == 0:
                continue
            
            buffer = self.class_buffers_X[c]
            if size < self.buffer_size:
                X_c = buffer[:size].copy()
            else:
                ptr = self.class_buffers_ptr[c]
                X_c = np.roll(buffer, -ptr, axis=0).copy()
            
            y_c = np.full(size, c, dtype=np.int32)
            X_all_list.append(X_c)
            y_all_list.append(y_c)
        
        if X_all_list:
            X_all = np.vstack(X_all_list)
            y_all = np.concatenate(y_all_list)
            
            final_fitness = numba_evaluate_population_overall(
                self.population, X_all, y_all, self.n_classes
            )
            best_idx = np.argmax(final_fitness)
            self.best_chromosome = self.population[best_idx].copy()
        else:
            # fallback
            self.best_chromosome = self.population[0].copy()
        
        self.training_time += time.time() - start_time
        self.n_updates += 1
        return self
    
    def predict(self, X):
        """Predict class labels (same decision rule as non-numba)."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_chromosome is None:
            return np.zeros(len(X), dtype=np.int32)
        return numba_predict(X, self.best_chromosome)
    
    def predict_proba(self, X):
        """Predict probabilities (extra convenience compared to non-numba)."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_chromosome is None:
            return np.ones((len(X), max(1, self.n_classes))) / max(1, self.n_classes)
        
        scores = X @ self.best_chromosome.T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


# =====================================================================
# ULTRA-FAST STANDARD DE (NO WAVES)
# =====================================================================

class UltraFastStandardDE:
    """
    Ultra-Fast Standard DE without waves.
    Behavior-matched to StandardDE (no waves, overall BA fitness).
    """
    
    def __init__(
        self,
        n_features,
        population_size=25,
        generations=24,       # match StandardDE.generations default
        F=0.8,
        CR=0.9,
        buffer_size=300,      # match buffer_size_per_class=200
        seed=None,
    ):
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.buffer_size = buffer_size
        
        if seed is not None:
            np.random.seed(seed)
        
        # Population and best
        self.population = None  # (pop_size, n_classes, n_features)
        self.best_chromosome = None
        
        # Class management
        self.n_classes = 0
        self.seen_classes = []
        
        # Per-class buffers
        self.class_buffers_X = {}
        self.class_buffers_ptr = {}
        self.class_buffers_size = {}
        
        # Stats
        self.training_time = 0.0
        self.n_updates = 0
    
    def _initialize_population(self):
        """Initialize population like StandardDE/StreamingDE_Waves."""
        self.population = (
            np.random.randn(
                self.population_size, self.n_classes, self.n_features
            ).astype(np.float32)
            * 0.1
        )
    
    def _update_class_buffer(self, c, X_c):
        """Update per-class circular buffer (same semantics as sliding window)."""
        if len(X_c) == 0:
            return
        
        if c not in self.class_buffers_X:
            self.class_buffers_X[c] = np.zeros(
                (self.buffer_size, self.n_features), dtype=np.float32
            )
            self.class_buffers_ptr[c] = 0
            self.class_buffers_size[c] = 0
        
        buffer = self.class_buffers_X[c]
        ptr = self.class_buffers_ptr[c]
        size = self.class_buffers_size[c]
        n_samples = len(X_c)
        
        if n_samples >= self.buffer_size:
            buffer[:] = X_c[-self.buffer_size:]
            self.class_buffers_size[c] = self.buffer_size
            self.class_buffers_ptr[c] = 0
        else:
            end_ptr = ptr + n_samples
            if end_ptr <= self.buffer_size:
                buffer[ptr:end_ptr] = X_c
            else:
                overflow = end_ptr - self.buffer_size
                buffer[ptr:] = X_c[: self.buffer_size - ptr]
                buffer[:overflow] = X_c[self.buffer_size - ptr :]
            
            self.class_buffers_ptr[c] = end_ptr % self.buffer_size
            self.class_buffers_size[c] = min(size + n_samples, self.buffer_size)
    
    def _sample_from_buffers(self, n_per_class):
        """
        Sample balanced batch from all classes.
        
        Mirrors StandardDE._construct_balanced_batch with:
        - batch_size = 250
        - per_class = n_per_class = batch_size // n_classes_in_buffer
        - up to n_per_class per class, without replacement.
        """
        X_list = []
        y_list = []
        
        for c in self.seen_classes:
            if c not in self.class_buffers_X:
                continue
            
            size = self.class_buffers_size[c]
            if size == 0:
                continue
            
            n = min(n_per_class, size)
            indices = np.random.choice(size, size=n, replace=False)
            
            buffer = self.class_buffers_X[c]
            if size < self.buffer_size:
                X_c = buffer[indices].copy()
            else:
                ptr = self.class_buffers_ptr[c]
                actual_indices = (ptr + indices) % self.buffer_size
                X_c = buffer[actual_indices].copy()
            
            X_list.append(X_c)
            y_list.append(np.full(n, c, dtype=np.int32))
        
        if not X_list:
            return None, None
        
        X_batch = np.vstack(X_list)
        y_batch = np.concatenate(y_list)
        
        # Shuffle
        perm = np.random.permutation(len(X_batch))
        return X_batch[perm], y_batch[perm]
    
    def partial_fit(self, X, y):
        """
        Online update without waves.
        
        Mirrors StandardDE.partial_fit:
        - Update/expand buffers.
        - Build a single balanced batch from all classes.
        - Run DE for a fixed number of generations (self.generations).
        - Select best chromosome by overall BA on all buffered data.
        """
        start_time = time.time()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        # ---- detect new classes and expand population ----
        unique_classes = np.unique(y)
        new_classes = set(unique_classes) - set(self.seen_classes)
        
        if new_classes:
            old_n_classes = self.n_classes
            
            for c in sorted(new_classes):
                self.seen_classes.append(int(c))
            self.seen_classes.sort()
            self.n_classes = len(self.seen_classes)
            
            if self.population is not None:
                old_pop = self.population
                new_pop = np.zeros(
                    (self.population_size, self.n_classes, self.n_features),
                    dtype=np.float32,
                )
                # keep old weights
                new_pop[:, :old_n_classes, :] = old_pop
                # random init for new class weights (match _expand_chromosomes)
                n_new = self.n_classes - old_n_classes
                if n_new > 0:
                    new_pop[:, old_n_classes:, :] = (
                        np.random.randn(
                            self.population_size, n_new, self.n_features
                        ).astype(np.float32)
                        * 0.1
                    )
                self.population = new_pop
        else:
            if self.n_classes == 0:
                self.seen_classes = sorted(int(c) for c in unique_classes)
                self.n_classes = len(self.seen_classes)
        
        if self.population is None and self.n_classes > 0:
            self._initialize_population()
        
        # ---- update buffers ----
        for c in unique_classes:
            c = int(c)
            X_c = X[y == c]
            self._update_class_buffer(c, X_c)
        
        # No classes with buffers yet?
        if len(self.class_buffers_X) == 0:
            return self
        
        # ---- construct balanced batch (same as StandardDE._construct_balanced_batch) ----
        batch_size = 250
        n_classes_in_buffer = len(self.class_buffers_X)
        n_per_class = max(1, batch_size // n_classes_in_buffer)
        
        X_train, y_train = self._sample_from_buffers(n_per_class=n_per_class)
        if X_train is None or len(X_train) == 0:
            # Fall back to current batch if buffers too small
            X_train, y_train = X, y
        
        # ---- DE evolution: fixed number of generations (no adaptivity) ----
        for _ in range(self.generations):
            new_population = np.zeros_like(self.population)
            
            for idx in range(self.population_size):
                target = self.population[idx]
                
                # Mutation
                mutant = numba_de_mutation(self.population, idx, self.F)
                
                # Crossover
                trial = numba_de_crossover(target, mutant, self.CR)
                
                # Selection using overall balanced accuracy
                f_target = numba_fitness_overall(
                    target, X_train, y_train, self.n_classes
                )
                f_trial = numba_fitness_overall(
                    trial, X_train, y_train, self.n_classes
                )
                
                if f_trial >= f_target:
                    new_population[idx] = trial
                else:
                    new_population[idx] = target
            
            self.population = new_population
        
        # ---- choose best chromosome using ALL buffered data ----
        X_all_list = []
        y_all_list = []
        for c in self.seen_classes:
            if c not in self.class_buffers_X:
                continue
            size = self.class_buffers_size[c]
            if size == 0:
                continue
            
            buffer = self.class_buffers_X[c]
            if size < self.buffer_size:
                X_c = buffer[:size].copy()
            else:
                ptr = self.class_buffers_ptr[c]
                X_c = np.roll(buffer, -ptr, axis=0).copy()
            
            y_c = np.full(size, c, dtype=np.int32)
            X_all_list.append(X_c)
            y_all_list.append(y_c)
        
        if X_all_list:
            X_all = np.vstack(X_all_list)
            y_all = np.concatenate(y_all_list)
            
            final_fitness = numba_evaluate_population_overall(
                self.population, X_all, y_all, self.n_classes
            )
            best_idx = np.argmax(final_fitness)
            self.best_chromosome = self.population[best_idx].copy()
        else:
            self.best_chromosome = self.population[0].copy()
        
        self.training_time += time.time() - start_time
        self.n_updates += 1
        return self
    
    def predict(self, X):
        """Predict class labels."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_chromosome is None:
            return np.zeros(len(X), dtype=np.int32)
        return numba_predict(X, self.best_chromosome)
    
    def predict_proba(self, X):
        """Predict probabilities (extra convenience)."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_chromosome is None:
            return np.ones((len(X), max(1, self.n_classes))) / max(1, self.n_classes)
        
        scores = X @ self.best_chromosome.T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
