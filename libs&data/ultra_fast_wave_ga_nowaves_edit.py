"""
Ultra-Fast WaveGA NoWaves Variant with Numba JIT Compilation
This variant uses blended fitness function and buffer but trains on all classes at once (no waves)

Prerequisites:
    pip install numba
"""

import time
import numpy as np
from numba import jit, prange
import warnings

warnings.filterwarnings('ignore')


# =====================================================================
# NUMBA-ACCELERATED CORE FUNCTIONS
# =====================================================================

@jit(nopython=True, cache=True, fastmath=True)
def numba_balanced_accuracy_binary(y_true, y_pred):
    """Ultra-fast binary balanced accuracy."""
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
def numba_balanced_accuracy_multiclass(y_true, y_pred, n_classes):
    """Ultra-fast multiclass balanced accuracy."""
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


@jit(nopython=True, cache=True, parallel=True, fastmath=True)
def numba_evaluate_population_blended(weights_3d, X, y, n_classes, alpha):
    """
    Fully parallelized population fitness evaluation with blended fitness.
    No target class - trains on all classes at once.
    
    Args:
        weights_3d: (pop_size, n_classes, n_features)
        X: (n_samples, n_features)
        y: (n_samples,)
        n_classes: int
        alpha: float (blending weight for per-class average)
    
    Returns:
        fitnesses: (pop_size,)
    """
    pop_size = weights_3d.shape[0]
    n_samples = X.shape[0]
    
    fitnesses = np.zeros(pop_size, dtype=np.float32)
    
    # Parallel over population
    for i in prange(pop_size):
        # Compute predictions for this chromosome
        y_pred = np.zeros(n_samples, dtype=np.int32)
        
        for s in range(n_samples):
            max_score = -1e10
            max_class = 0
            
            for c in range(n_classes):
                score = 0.0
                for f in range(X.shape[1]):
                    score += weights_3d[i, c, f] * X[s, f]
                
                if score > max_score:
                    max_score = score
                    max_class = c
            
            y_pred[s] = max_class
        
        # Per-class BA (average of all class-specific BAs)
        class_bas = np.zeros(n_classes, dtype=np.float32)
        
        for c in range(n_classes):
            y_binary = np.zeros(n_samples, dtype=np.int32)
            y_pred_binary = np.zeros(n_samples, dtype=np.int32)
            
            for s in range(n_samples):
                y_binary[s] = (y[s] == c)
                y_pred_binary[s] = (y_pred[s] == c)
            
            class_bas[c] = numba_balanced_accuracy_binary(y_binary, y_pred_binary)
        
        f_class_avg = np.mean(class_bas)
        
        # Overall multiclass BA
        f_overall = numba_balanced_accuracy_multiclass(y, y_pred, n_classes)
        
        # Blended fitness
        fitnesses[i] = alpha * f_class_avg + (1.0 - alpha) * f_overall
    
    return fitnesses


@jit(nopython=True, cache=True, fastmath=True)
def numba_arithmetic_crossover(weights1, weights2, alpha):
    """Vectorized arithmetic crossover."""
    return alpha * weights1 + (1.0 - alpha) * weights2


@jit(nopython=True, cache=True, fastmath=True)
def numba_uniform_crossover(weights1, weights2, mask):
    """Vectorized uniform crossover."""
    n_classes, n_features = weights1.shape
    child = np.empty_like(weights1)
    
    for i in range(n_classes):
        for j in range(n_features):
            if mask[i, j]:
                child[i, j] = weights1[i, j]
            else:
                child[i, j] = weights2[i, j]
    
    return child


@jit(nopython=True, cache=True, fastmath=True)
def numba_mutate(weights, mutation_mask, mutation_values):
    """In-place mutation."""
    n_classes, n_features = weights.shape
    
    for i in range(n_classes):
        for j in range(n_features):
            if mutation_mask[i, j]:
                weights[i, j] += mutation_values[i, j]


@jit(nopython=True, cache=True)
def numba_predict(X, weights):
    """Fast batch prediction."""
    n_samples = X.shape[0]
    n_classes = weights.shape[0]
    
    y_pred = np.zeros(n_samples, dtype=np.int32)
    
    for s in range(n_samples):
        max_score = -1e10
        max_class = 0
        
        for c in range(n_classes):
            score = 0.0
            for f in range(X.shape[1]):
                score += weights[c, f] * X[s, f]
            
            if score > max_score:
                max_score = score
                max_class = c
        
        y_pred[s] = max_class
    
    return y_pred


# =====================================================================
# MAIN ULTRA-FAST WAVEGA NOWAVES CLASS
# =====================================================================

class UltraFastWaveGANoWaves:
    """
    Ultra-optimized WaveGA NoWaves variant with Numba JIT compilation.
    Uses blended fitness function and buffer but trains on all classes at once (no waves).
    """
    
    def __init__(self,
                 n_features,
                 population_size=30,
                 generations_per_wave=10,  # To match WaveGA parameter naming
                 n_cycles=3,  # To match WaveGA parameter naming
                 mutation_rate=0.1,
                 mutation_strength=0.1,
                 buffer_size=1000,
                 alpha=0.7,
                 crossover_type="arithmetic",
                 seed=None):
        
        if seed is not None:
            np.random.seed(seed)
        
        self.n_features = n_features
        self.population_size = population_size
        self.generations_per_wave = generations_per_wave
        self.n_cycles = n_cycles
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.crossover_type = crossover_type
        
        # Internal state
        self.n_classes = 0
        self.seen_classes = []
        self.population_weights = None
        self.population_fitness = None
        self.best_weights = None
        
        # Class buffers
        self.class_buffers_X = {}
        self.class_buffers_ptr = {}
        self.class_buffers_size = {}
        
        # Statistics
        self.training_time = 0.0
        self.n_updates = 0
        
        # Random state for reproducibility
        self._random_state = np.random.RandomState(seed)
    
    def _get_random_uniform(self):
        """Thread-safe random number."""
        return self._random_state.rand()
    
    def _initialize_population(self, X, y):
        """Initialize population with small random weights."""
        self.population_weights = np.random.randn(
            self.population_size, self.n_classes, self.n_features
        ).astype(np.float32) * 0.01
        
        self.population_fitness = np.zeros(self.population_size, dtype=np.float32)
    
    def _update_class_buffer(self, c, X_c):
        """Update circular buffer for class c."""
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
            # Replace entire buffer
            buffer[:] = X_c[-self.buffer_size:]
            self.class_buffers_size[c] = self.buffer_size
            self.class_buffers_ptr[c] = 0
        else:
            # Add circularly
            end_ptr = ptr + n_samples
            if end_ptr <= self.buffer_size:
                buffer[ptr:end_ptr] = X_c
            else:
                overflow = end_ptr - self.buffer_size
                buffer[ptr:] = X_c[:self.buffer_size - ptr]
                buffer[:overflow] = X_c[self.buffer_size - ptr:]
            
            self.class_buffers_ptr[c] = end_ptr % self.buffer_size
            self.class_buffers_size[c] = min(size + n_samples, self.buffer_size)
    
    def _sample_from_buffers(self, n_per_class=50):
        """Sample from all class buffers to create balanced training batch."""
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
            ptr = self.class_buffers_ptr[c]
            
            if size < self.buffer_size:
                X_c = buffer[indices].copy()
            else:
                # Adjust for circular buffer
                actual_indices = (ptr + indices) % self.buffer_size
                X_c = buffer[actual_indices].copy()
            
            y_c = np.full(len(X_c), c, dtype=np.int32)
            X_list.append(X_c)
            y_list.append(y_c)
        
        if not X_list:
            return None, None
        
        X_batch = np.vstack(X_list)
        y_batch = np.concatenate(y_list)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_batch))
        return X_batch[shuffle_idx], y_batch[shuffle_idx]
    
    def _train_generation(self, X, y):
        """Single generation of training on all classes."""
        # Evaluate population using Numba
        self.population_fitness = numba_evaluate_population_blended(
            self.population_weights, X, y, self.n_classes, self.alpha
        )
        
        # Sort by fitness
        sorted_indices = np.argsort(self.population_fitness)[::-1]
        
        # Elitism: keep top 2
        new_population = np.zeros_like(self.population_weights)
        new_population[0] = self.population_weights[sorted_indices[0]].copy()
        new_population[1] = self.population_weights[sorted_indices[1]].copy()
        
        # Generate offspring
        offspring_idx = 2
        while offspring_idx < self.population_size:
            # Tournament selection
            t1_idx = np.random.choice(self.population_size, size=3, replace=False)
            t2_idx = np.random.choice(self.population_size, size=3, replace=False)
            
            parent1_idx = t1_idx[np.argmax(self.population_fitness[t1_idx])]
            parent2_idx = t2_idx[np.argmax(self.population_fitness[t2_idx])]
            
            parent1 = self.population_weights[parent1_idx]
            parent2 = self.population_weights[parent2_idx]
            
            # Crossover
            if self.crossover_type == "arithmetic":
                alpha_cross = self._get_random_uniform()
                alpha_cross = 0.3 + 0.4 * alpha_cross  # Map to [0.3, 0.7]
                child = numba_arithmetic_crossover(parent1, parent2, alpha_cross)
            else:  # uniform
                mask = np.random.rand(self.n_classes, self.n_features) < 0.5
                child = numba_uniform_crossover(parent1, parent2, mask)
            
            # Mutation
            if self._get_random_uniform() < self.mutation_rate:
                mutation_mask = np.random.rand(self.n_classes, self.n_features) < 0.1
                mutation_values = np.random.randn(self.n_classes, self.n_features) * self.mutation_strength
                numba_mutate(child, mutation_mask, mutation_values)
            
            new_population[offspring_idx] = child
            offspring_idx += 1
        
        self.population_weights = new_population
    
    def partial_fit(self, X, y):
        """Ultra-fast incremental update without waves."""
        start_time = time.time()
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int32)
        
        # Detect new classes
        unique_classes = np.unique(y)
        new_classes = set(unique_classes) - set(self.seen_classes)
        
        if new_classes:
            print(f"NEW CLASSES: {new_classes}")
            old_n_classes = self.n_classes
            
            for c in sorted(new_classes):
                self.seen_classes.append(c)
            self.seen_classes.sort()
            self.n_classes = len(self.seen_classes)
            
            # Expand population
            if self.population_weights is not None:
                old_pop = self.population_weights
                new_pop = np.zeros(
                    (self.population_size, self.n_classes, self.n_features),
                    dtype=np.float32
                )
                new_pop[:, :old_n_classes, :] = old_pop
                self.population_weights = new_pop
        else:
            if self.n_classes == 0:
                self.seen_classes = sorted(unique_classes)
                self.n_classes = len(self.seen_classes)
        
        # Initialize if needed
        if self.population_weights is None:
            self._initialize_population(X, y)
        
        # Update buffers
        for c in unique_classes:
            X_c = X[y == c]
            self._update_class_buffer(c, X_c)
        
        # Sample from buffers for training
        X_train, y_train = self._sample_from_buffers(n_per_class=150)
        
        if X_train is None or len(X_train) < 10:
            # Not enough data yet, use current batch
            X_train, y_train = X, y
        
        # FAIR COMPARISON: Match WaveGA's total computational budget
        # WaveGA does: generations_per_wave × n_cycles × n_classes generations
        # StandardGA should do: generations_per_wave × n_cycles × n_classes generations too
        total_generations = self.generations_per_wave * self.n_cycles * self.n_classes
        
        # Adaptive reduction for early training
        total_buffer_size = sum(self.class_buffers_size.values())
        if total_buffer_size < 200:
            effective_gens = max(self.generations_per_wave, total_generations // 3)
        elif total_buffer_size < 500:
            effective_gens = max(self.generations_per_wave * 2, total_generations // 2)
        else:
            effective_gens = total_generations
        
        # Train for multiple generations on all classes at once
        for _ in range(effective_gens):
            self._train_generation(X_train, y_train)
        
        # Update best
        final_fitness = numba_evaluate_population_blended(
            self.population_weights, X_train, y_train, self.n_classes, self.alpha
        )
        best_idx = np.argmax(final_fitness)
        self.best_weights = self.population_weights[best_idx].copy()
        
        self.training_time += time.time() - start_time
        self.n_updates += 1
    
    def predict(self, X):
        """Fast prediction."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_weights is None:
            return np.zeros(len(X), dtype=np.int32)
        return numba_predict(X, self.best_weights)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        X = np.asarray(X, dtype=np.float32)
        if self.best_weights is None:
            return np.ones((len(X), max(1, self.n_classes))) / max(1, self.n_classes)
        
        scores = X @ self.best_weights.T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
