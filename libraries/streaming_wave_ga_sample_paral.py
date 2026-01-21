"""
Wave-Based Genetic Algorithm for Incremental Multiclass Classification
Handles sequential class arrival, concept drift, and class imbalance
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore', message='.*single label.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class Chromosome:
    """Linear classifier chromosome: W matrix of shape (n_classes, n_features)"""
    
    def __init__(self, n_classes, n_features):
        self.n_classes = n_classes
        self.n_features = n_features
        self.weights = np.random.randn(n_classes, n_features) * 0.1
        self.fitness = 0.0
        
    def predict(self, X):
        """Predict class labels"""
        scores = X @ self.weights.T  # (n_samples, n_classes)
        return np.argmax(scores, axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities (softmax)"""
        scores = X @ self.weights.T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def copy(self):
        """Create deep copy"""
        new = Chromosome(self.n_classes, self.n_features)
        new.weights = self.weights.copy()
        new.fitness = self.fitness
        return new
class StandardOnlineGA:
    """
    Standard GA without waves - FIXED for class expansion
    Vectorized fitness evaluation over the whole population.
    """
    
    def __init__(self, n_features, population_size=30, generations=30,
                 mutation_rate=0.1, mutation_strength=0.1):
        
        self.n_features = n_features
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        self.population = []
        self.best_chromosome = None
        self.n_classes = 0
        self.seen_classes = set()
        self.training_time = 0.0

    # ------------ vectorized helpers ------------

    def _stack_weights(self):
        """
        Stack all chromosome weights into a single array of shape
        (pop_size, n_classes, n_features).
        """
        return np.stack([chrom.weights for chrom in self.population], axis=0)

    def _population_predictions(self, X):
        """
        Vectorized prediction for all chromosomes at once.

        Returns:
            y_pred_all: (pop_size, n_samples) int labels
        """
        # W: (P, C, F), X: (N, F)
        W = self._stack_weights()                      # (P, C, F)
        scores = np.einsum('nf,pcf->pnc', X, W)       # (P, N, C)
        y_pred_all = np.argmax(scores, axis=2)        # (P, N)
        return y_pred_all

    def _population_balanced_accuracy(self, y_true, y_pred_all):
        """
        Vectorized multiclass balanced accuracy for all chromosomes.

        y_true      : (N,)
        y_pred_all  : (P, N)
        returns ba  : (P,)
        """
        y_true = np.asarray(y_true)
        y_pred_all = np.asarray(y_pred_all)
        P, N = y_pred_all.shape

        classes = np.unique(y_true)
        if classes.size < 2:
            # Fall back to simple accuracy if only one class present
            acc = (y_pred_all == y_true[None, :]).mean(axis=1)
            return acc

        # y_true_onehot: (K, N)
        K = classes.size
        y_true_onehot = (y_true[None, :] == classes[:, None])

        # y_pred_onehot: (P, K, N)
        y_pred_onehot = (y_pred_all[:, None, :] == classes[None, :, None])

        # tp: (P, K)
        tp = np.sum(y_pred_onehot & y_true_onehot[None, :, :], axis=2)
        # fn: (P, K)
        fn = np.sum((~y_pred_onehot) & y_true_onehot[None, :, :], axis=2)

        denom = tp + fn
        recall = np.divide(
            tp,
            denom,
            out=np.zeros_like(tp, dtype=float),
            where=denom > 0
        )

        # mean recall across classes
        ba = recall.mean(axis=1)   # (P,)
        return ba

    def _evaluate_population_fitness(self, X, y):
        """
        Compute fitness for the whole population in a vectorized way.
        Uses balanced accuracy when multiple classes are present,
        otherwise standard accuracy.
        """
        y = np.asarray(y)
        y_pred_all = self._population_predictions(X)          # (P, N)
        fitness = self._population_balanced_accuracy(y, y_pred_all)  # (P,)
        return fitness

    # ------------ training ------------

    def partial_fit(self, X, y):
        """Standard GA evolution - FIXED for new classes, vectorized fitness."""
        start_time = time.time()
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Detect new classes
        new_classes = set(np.unique(y)) - self.seen_classes
        
        if new_classes:
            # Class expansion!
            old_n_classes = self.n_classes
            self.n_classes = len(self.seen_classes | set(np.unique(y)))
            self.seen_classes.update(np.unique(y))
            
            # Expand all existing chromosomes
            for chrom in self.population:
                old_weights = chrom.weights
                chrom.weights = np.zeros((self.n_classes, self.n_features))
                chrom.weights[:old_n_classes] = old_weights
                chrom.n_classes = self.n_classes
            
            # Expand best chromosome too
            if self.best_chromosome is not None:
                old_weights = self.best_chromosome.weights
                self.best_chromosome.weights = np.zeros((self.n_classes, self.n_features))
                self.best_chromosome.weights[:old_n_classes] = old_weights
                self.best_chromosome.n_classes = self.n_classes
        else:
            self.seen_classes.update(np.unique(y))
            if self.n_classes == 0:
                self.n_classes = len(self.seen_classes)
        
        # Initialize population if needed
        if len(self.population) == 0:
            for _ in range(self.population_size):
                chrom = Chromosome(self.n_classes, self.n_features)
                self.population.append(chrom)
        
        # Skip training if too few samples
        if len(X) < 20:
            self.training_time += time.time() - start_time
            return
        
        # Standard GA evolution with vectorized fitness
        for gen in range(self.generations):
            # Vectorized fitness for whole population
            fitness = self._evaluate_population_fitness(X, y)   # (P,)
            for chrom, fit in zip(self.population, fitness):
                chrom.fitness = fit
            
            # Selection, crossover, mutation
            sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            new_population = [c.copy() for c in sorted_pop[:2]]  # Elitism
            
            while len(new_population) < self.population_size:
                # Tournament selection
                p1 = max(np.random.choice(self.population, 3, replace=False), 
                         key=lambda c: c.fitness)
                p2 = max(np.random.choice(self.population, 3, replace=False), 
                         key=lambda c: c.fitness)
                
                # Crossover - SAFE version
                child = Chromosome(self.n_classes, self.n_features)
                mask = np.random.rand(self.n_classes, self.n_features) < 0.5
                child.weights = np.where(mask, p1.weights, p2.weights)
                
                # Mutation
                if np.random.rand() < self.mutation_rate:
                    mutation = np.random.randn(self.n_classes, self.n_features) * self.mutation_strength
                    child.weights += mutation
                
                new_population.append(child)
            
            self.population = new_population
        
        # Update best
        self.best_chromosome = max(self.population, key=lambda c: c.fitness).copy()
        self.training_time += time.time() - start_time
    
    def predict(self, X):
        if self.best_chromosome is None:
            return np.zeros(len(X), dtype=int)
        return self.best_chromosome.predict(X)
class perSampleStreamingWaveGA:
    """
    Wave-Based GA for streaming data with sequential class arrival
    """
    
    def __init__(self, n_features, population_size=30, 
                 generations_per_wave=10, n_cycles=3,
                 mutation_rate=0.1, mutation_strength=0.1):
        
        self.n_features = n_features
        self.population_size = population_size
        self.generations_per_wave = generations_per_wave
        self.n_cycles = n_cycles
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        
        self.population = []
        self.best_chromosome = None
        self.seen_classes = set()
        self.n_classes = 0
        
        # Buffers for incremental learning
        self.class_buffers = defaultdict(list)
        self.buffer_size = 1000
        
        # Statistics
        self.training_time = 0.0
        self.n_updates = 0
        
    def _initialize_population(self, X, y):
        """Class-aware initialization using actual samples"""
        self.population = []
        classes = np.unique(y)
        
        # Create diverse population
        for i in range(self.population_size):
            chrom = Chromosome(self.n_classes, self.n_features)
            
            # Initialize weights from class samples (30% of population)
            if i < self.population_size * 0.3:
                for c in classes:
                    X_c = X[y == c]
                    if len(X_c) > 0:
                        sample = X_c[np.random.randint(len(X_c))]
                        chrom.weights[c] = sample / (np.linalg.norm(sample) + 1e-8)
            
            self.population.append(chrom)
    
    def _evaluate_fitness_for_class(self, chromosome, X, y, target_class):
        """One-vs-rest fitness for specific class"""
        y_binary = (y == target_class).astype(int)
        y_pred = chromosome.predict(X)
        y_pred_binary = (y_pred == target_class).astype(int)
        
        # Balanced accuracy for this class
        if len(np.unique(y_binary)) < 2:
            return 0.5
        
        from sklearn.metrics import balanced_accuracy_score
        return balanced_accuracy_score(y_binary, y_pred_binary)
    
    def _evaluate_overall_fitness(self, chromosome, X, y):
        """Overall balanced accuracy across all classes"""
        y_pred = chromosome.predict(X)
        return balanced_accuracy_score(y, y_pred)
    
    def _tournament_selection(self, k=3):
        """Select parent via tournament"""
        tournament = np.random.choice(self.population, size=k, replace=False)
        return max(tournament, key=lambda c: c.fitness)
    
    def _crossover(self, parent1, parent2):
        """Uniform crossover"""
        child = Chromosome(self.n_classes, self.n_features)
        mask = np.random.rand(self.n_classes, self.n_features) < 0.5
        child.weights = np.where(mask, parent1.weights, parent2.weights)
        return child
    
    def _mutate(self, chromosome):
        """Gaussian mutation"""
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.randn(self.n_classes, self.n_features) * self.mutation_strength
            chromosome.weights += mutation
    
    def _wave_training(self, X, y, target_class):
        """Execute wave training for specific class"""
        
        for generation in range(self.generations_per_wave):
            if generation % 5 == 0:
                print('.', end='', flush=True)
            # Evaluate fitness focused on target class
            for chrom in self.population:
                class_fitness = self._evaluate_fitness_for_class(chrom, X, y, target_class)
                overall_fitness = self._evaluate_overall_fitness(chrom, X, y)
                # Blend: 70% class-focused, 30% overall
                chrom.fitness = 0.7 * class_fitness + 0.3 * overall_fitness
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best 2
            sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            new_population.extend([c.copy() for c in sorted_pop[:2]])
            
            # Generate rest via crossover + mutation
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)
            
            self.population = new_population

    # NEW: helper to ensure classes / expand chromosomes
    def _ensure_classes(self, y):
        """
        Ensure internal structures account for all labels in y.
        Expands chromosomes when new classes appear.
        """
        labels = np.unique(y)
        new_classes = set(labels) - self.seen_classes

        if new_classes:
            old_n_classes = self.n_classes
            # Update seen_classes and class count
            self.seen_classes.update(labels)
            self.n_classes = len(self.seen_classes)

            # Expand existing chromosomes
            for chrom in self.population:
                old_weights = chrom.weights
                chrom.weights = np.zeros((self.n_classes, self.n_features))
                if old_n_classes > 0:
                    chrom.weights[:old_n_classes] = old_weights
                chrom.n_classes = self.n_classes
        else:
            # Just keep track of them
            self.seen_classes.update(labels)
            if self.n_classes == 0:
                self.n_classes = len(self.seen_classes)

    def partial_fit(self, X, y):
        """
        Incremental update with new batch - OPTIMIZED VERSION
        """
        start_time = time.time()
        
        # Use shared class-expansion logic (NEW)
        self._ensure_classes(y)
        
        # Initialize population if first time
        if len(self.population) == 0:
            self._initialize_population(X, y)
        
        # Update buffers
        for c in np.unique(y):
            X_c = X[y == c]
            self.class_buffers[c].extend(X_c.tolist())
            if len(self.class_buffers[c]) > self.buffer_size:
                self.class_buffers[c] = self.class_buffers[c][-self.buffer_size:]
        
        # OPTIMIZATION 2: Adaptive cycles based on data
        total_buffer_size = sum(len(buf) for buf in self.class_buffers.values())
        if total_buffer_size < 200:
            effective_cycles = 1  # Fast for early stages
        elif total_buffer_size < 500:
            effective_cycles = 2
        else:
            effective_cycles = self.n_cycles
        
        # OPTIMIZATION 3: Adaptive generations based on number of classes
        n_active_classes = len([c for c in self.seen_classes if c in self.class_buffers and len(self.class_buffers[c]) > 10])
        if n_active_classes <= 1:
            effective_generations = max(3, self.generations_per_wave // 2)
        else:
            effective_generations = self.generations_per_wave
        
        # Execute wave training
        classes = sorted(self.seen_classes)
        for cycle in range(effective_cycles):
            for target_class in classes:
                if target_class not in self.class_buffers or len(self.class_buffers[target_class]) < 10:
                    continue  # Skip classes with insufficient data
                
                # Sample from buffer (smaller samples for speed)
                buffer_size = len(self.class_buffers[target_class])
                n_samples = min(150, buffer_size)  # Was 200, now 150
                
                X_wave = []
                y_wave = []
                
                # Target class
                indices = np.random.choice(buffer_size, n_samples // 2, replace=False)
                X_wave.extend([self.class_buffers[target_class][i] for i in indices])
                y_wave.extend([target_class] * (n_samples // 2))
                
                # Other classes
                other_classes = [c for c in classes if c != target_class and c in self.class_buffers]
                if other_classes:
                    for c in other_classes:
                        if len(self.class_buffers[c]) > 0:
                            n_c = (n_samples // 2) // len(other_classes)
                            indices = np.random.choice(len(self.class_buffers[c]), 
                                                      min(n_c, len(self.class_buffers[c])), 
                                                      replace=False)
                            X_wave.extend([self.class_buffers[c][i] for i in indices])
                            y_wave.extend([c] * len(indices))
                
                if len(X_wave) > 10:
                    X_wave = np.array(X_wave)
                    y_wave = np.array(y_wave)
                    
                    # Wave training with adaptive generations
                    self._wave_training_fast(X_wave, y_wave, target_class, effective_generations)
        
        # Update best
        for chrom in self.population:
            chrom.fitness = self._evaluate_overall_fitness(chrom, X, y)
        self.best_chromosome = max(self.population, key=lambda c: c.fitness)
        
        self.training_time += time.time() - start_time
        self.n_updates += 1

    def _wave_training_fast(self, X, y, target_class, n_generations):
        """Fast wave training with reduced generations"""
        
        for generation in range(n_generations):
            # Evaluate
            for chrom in self.population:
                class_fitness = self._evaluate_fitness_for_class(chrom, X, y, target_class)
                overall_fitness = self._evaluate_overall_fitness(chrom, X, y)
                chrom.fitness = 0.7 * class_fitness + 0.3 * overall_fitness
            
            # Next generation
            new_population = []
            sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            new_population.extend([c.copy() for c in sorted_pop[:2]])  # Elitism
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)
            
            self.population = new_population

    # NEW: micro-evolution step for single-sample updates
    def _micro_evolution(self, X, y):
        """
        Extremely lightweight GA update for a tiny batch (often size 1).
        Uses the same class-focused + overall fitness blend.
        """
        target_class = y[0]

        # Evaluate fitness
        for chrom in self.population:
            class_fitness = self._evaluate_fitness_for_class(chrom, X, y, target_class)
            overall_fitness = self._evaluate_overall_fitness(chrom, X, y)
            chrom.fitness = 0.7 * class_fitness + 0.3 * overall_fitness

        # One generation: elitism + crossover + mutation
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        new_population = [c.copy() for c in sorted_pop[:2]]  # elitism

        while len(new_population) < self.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._crossover(p1, p2)
            self._mutate(child)
            new_population.append(child)

        self.population = new_population

    # NEW: true per-sample online update
    def partial_fit_single(self, x, y):
        """
        Incremental update with a single sample: (x, y).
        This performs a very cheap micro-evolution step.
        """
        start_time = time.time()

        # Make (1, n_features) and (1,)
        x = np.asarray(x).reshape(1, -1)
        y = np.asarray([y])

        # Ensure classes and expand chromosomes if needed
        self._ensure_classes(y)

        # Initialize population on first sample
        if len(self.population) == 0:
            self._initialize_population(x, y)
            self.best_chromosome = self.population[0]

        # Update buffer for this class
        label = y[0]
        self.class_buffers[label].append(x[0].tolist())
        if len(self.class_buffers[label]) > self.buffer_size:
            self.class_buffers[label] = self.class_buffers[label][-self.buffer_size:]

        # Perform one micro-evolution step
        self._micro_evolution(x, y)

        # Update best chromosome (using the single sample)
        for chrom in self.population:
            chrom.fitness = self._evaluate_overall_fitness(chrom, x, y)
        self.best_chromosome = max(self.population, key=lambda c: c.fitness)

        self.training_time += time.time() - start_time
        self.n_updates += 1

    def predict(self, X):
        """Predict using best chromosome"""
        if self.best_chromosome is None:
            return np.zeros(len(X), dtype=int)
        return self.best_chromosome.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.best_chromosome is None:
            n_samples = len(X)
            return np.ones((n_samples, max(1, self.n_classes))) / max(1, self.n_classes)
        return self.best_chromosome.predict_proba(X)
class newperSampleStreamingWaveGA:
    """
    Enhanced Wave-Based GA for streaming data with:
      - per-sample micro-evolution
      - class-imbalance handling
      - simple concept-drift detection
      - adaptive buffers and wave settings
    """

    def __init__(self, n_features,
                   population_size=30, 
                 generations_per_wave=10, 
                 n_cycles=3,
                 mutation_rate=0.1, 
                 mutation_strength=0.1,
                 # imbalance-related
                 class_weight_power=0.5,        # how strongly to upweight rare classes
                 majority_freq_threshold=0.6,   # freq above which a class is "majority"
                 majority_update_prob=0.2,      # prob to do per-sample update for majority
                 # buffer-related
                 base_buffer_size=1000,
                 min_buffer_size=200,
                 max_buffer_size=2000,
                 # drift-related
                 drift_window_size=30,
                 drift_min_window=10,
                 drift_drop_threshold=0.1,
                 drift_mutation_factor=2.0,
          
                drift_generations_factor=1.5,
                 drift_cooldown=50):
        
        self.n_features = n_features
        self.population_size = population_size
        self.generations_per_wave = generations_per_wave
        self.n_cycles = n_cycles
        
        # store "base" mutation params, we will change them dynamically under drift
        self.mutation_rate_base = mutation_rate
        self.mutation_strength_base = mutation_strength
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength

        # GA population
        self.population = []
        self.best_chromosome = None
        
        # class tracking
        self.seen_classes = set()
        self.n_classes = 0
        from collections import defaultdict
        self.class_buffers = defaultdict(list)
        self.class_counts = defaultdict(int)   # count how many samples seen per class
        
        # buffer sizing
        self.base_buffer_size = base_buffer_size
        self.min_buffer_size = min_buffer_size
        self.max_buffer_size = max_buffer_size
        
        # imbalance handling
        self.class_weight_power = class_weight_power
        self.majority_freq_threshold = majority_freq_threshold
        self.majority_update_prob = majority_update_prob
        
        # drift handling
        self.drift_window = []                 # recent BA history
        self.drift_window_size = drift_window_size
        self.drift_min_window = drift_min_window
        self.drift_drop_threshold = drift_drop_threshold
        self.drift_mutation_factor = drift_mutation_factor
        self.drift_generations_factor = drift_generations_factor
        self.drift_cooldown = drift_cooldown
        self.in_drift_mode = False
        self.drift_cooldown_counter = 0
        
        # statistics
        self.training_time = 0.0
        self.n_updates = 0

    # ===============================
    #  Internal helpers
    # ===============================

    def _initialize_population(self, X, y):
        """Class-aware initialization using actual samples."""
        import numpy as np

        self.population = []
        classes = np.unique(y)
        
        for i in range(self.population_size):
            chrom = Chromosome(self.n_classes, self.n_features)
            # initialize a subset from real samples
            if i < self.population_size * 0.3:
                for c in classes:
                    X_c = X[y == c]
                    if len(X_c) > 0:
                        sample = X_c[np.random.randint(len(X_c))]
                        norm = np.linalg.norm(sample) + 1e-8
                        chrom.weights[c] = sample / norm
            self.population.append(chrom)

    def _evaluate_fitness_for_class(self, chromosome, X, y, target_class):
        """
        One-vs-rest fitness for specific class, with optional
        class-frequency–based reweighting for imbalance.
        """
        import numpy as np
        from sklearn.metrics import balanced_accuracy_score

        y_binary = (y == target_class).astype(int)
        y_pred = chromosome.predict(X)
        y_pred_binary = (y_pred == target_class).astype(int)

        if len(np.unique(y_binary)) < 2:
            class_fitness = 0.5
        else:
            class_fitness = balanced_accuracy_score(y_binary, y_pred_binary)

        # imbalance-aware weighting: upweight rare classes
        w_c = self._get_class_weight(target_class)
        return class_fitness * w_c

    def _evaluate_overall_fitness(self, chromosome, X, y):
        """Overall balanced accuracy across all classes."""
        from sklearn.metrics import balanced_accuracy_score
        import numpy as np

        y_pred = chromosome.predict(X)
        if len(np.unique(y)) < 2:
            return np.mean(y_pred == y)
        return balanced_accuracy_score(y, y_pred)

    def _tournament_selection(self, k=3):
        """Select parent via tournament."""
        import numpy as np
        tournament = np.random.choice(self.population, size=k, replace=False)
        return max(tournament, key=lambda c: c.fitness)
    
    def _crossover(self, parent1, parent2):
        """Uniform crossover."""
        import numpy as np
        child = Chromosome(self.n_classes, self.n_features)
        mask = np.random.rand(self.n_classes, self.n_features) < 0.5
        child.weights = np.where(mask, parent1.weights, parent2.weights)
        return child
    
    def _mutate(self, chromosome):
        """Gaussian mutation using current mutation_rate / strength."""
        import numpy as np
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.randn(self.n_classes, self.n_features) * self.mutation_strength
            chromosome.weights += mutation

    # ---------- Class expansion / tracking ----------

    def _ensure_classes(self, y):
        """
        Ensure internal structures account for all labels in y.
        Expands chromosomes when new classes appear.
        """
        import numpy as np
        labels = np.unique(y)
        new_classes = set(labels) - self.seen_classes

        if new_classes:
            old_n_classes = self.n_classes
            self.seen_classes.update(labels)
            self.n_classes = len(self.seen_classes)

            # expand existing chromosomes
            for chrom in self.population:
                old_weights = chrom.weights
                chrom.weights = np.zeros((self.n_classes, self.n_features))
                if old_n_classes > 0:
                    chrom.weights[:old_n_classes] = old_weights
                chrom.n_classes = self.n_classes
        else:
            self.seen_classes.update(labels)
            if self.n_classes == 0:
                self.n_classes = len(self.seen_classes)

    def _update_class_counts(self, y):
        """Update global frequency counts for each class."""
        import numpy as np
        for c in np.unique(y):
            self.class_counts[int(c)] += int(np.sum(y == c))

    def _get_total_samples_seen(self):
        """Total number of samples processed so far."""
        return sum(self.class_counts.values())

    def _get_class_weight(self, c):
        """
        Compute imbalance-aware weight for class c.
        Smaller classes get higher weights.
        """
        import numpy as np
        total = self._get_total_samples_seen()
        if total == 0:
            return 1.0
        count_c = self.class_counts.get(int(c), 0)
        if count_c == 0:
            return 1.0
        freq_c = count_c / total
        # rarer class (small freq) -> larger weight
        w = (1.0 / (freq_c + 1e-8)) ** self.class_weight_power
        # normalize a bit so weights aren't extreme
        # cap between 1 and 5
        return float(np.clip(w, 1.0, 5.0))

    def _get_effective_buffer_size(self, c):
        """
        Choose per-class buffer size depending on how frequent the class is.
        Majority classes get smaller buffers (less memory),
        minority classes get larger buffers (more reuse).
        """
        total = self._get_total_samples_seen()
        if total == 0:
            return self.base_buffer_size

        count_c = self.class_counts.get(int(c), 0)
        if count_c == 0:
            return self.base_buffer_size

        freq_c = count_c / total
        # if more frequent than threshold ⇒ majority: smaller buffer
        if freq_c > self.majority_freq_threshold:
            size = int(self.base_buffer_size * 0.5)
        else:
            size = int(self.base_buffer_size * 1.5)

        size = max(self.min_buffer_size, min(self.max_buffer_size, size))
        return size

    # ---------- Drift handling ----------

    def _update_drift_state(self, X, y):
        """
        Track balanced accuracy over batches and enable/disable drift mode.
        In drift mode, mutation and generations per wave are boosted and
        part of the population may be reinitialized.
        """
        from sklearn.metrics import balanced_accuracy_score
        import numpy as np

        if self.best_chromosome is None:
            return

        if len(y) < 2 or len(np.unique(y)) < 2:
            return

        y_pred = self.best_chromosome.predict(X)
        ba = balanced_accuracy_score(y, y_pred)

        # update window
        self.drift_window.append(ba)
        if len(self.drift_window) > self.drift_window_size:
            self.drift_window.pop(0)

        # maybe enter drift mode
        if len(self.drift_window) >= self.drift_min_window:
            past = self.drift_window[:-1]
            if len(past) > 0:
                past_mean = float(np.mean(past))
                if (past_mean - ba) > self.drift_drop_threshold and not self.in_drift_mode:
                    # significant drop: enter drift mode
                    self.in_drift_mode = True
                    self.drift_cooldown_counter = self.drift_cooldown
                    # reinitialize worst half of population to increase exploration
                    if len(self.population) > 0:
                        sorted_pop = sorted(self.population, key=lambda c: c.fitness)
                        half = len(sorted_pop) // 2
                        for chrom in sorted_pop[:half]:
                            chrom.weights = np.random.randn(self.n_classes, self.n_features) * 0.1
                            chrom.fitness = 0.0

        # countdown drift mode
        if self.in_drift_mode:
            self.drift_cooldown_counter -= 1
            if self.drift_cooldown_counter <= 0:
                self.in_drift_mode = False

        # adjust mutation for next training based on drift state
        if self.in_drift_mode:
            self.mutation_rate = self.mutation_rate_base * self.drift_mutation_factor
            self.mutation_strength = self.mutation_strength_base * self.drift_mutation_factor
        else:
            self.mutation_rate = self.mutation_rate_base
            self.mutation_strength = self.mutation_strength_base

    # ===============================
    #  Main training routines
    # ===============================

    def _wave_training_fast(self, X, y, target_class, n_generations):
        """Fast wave training with reduced generations."""
        import numpy as np

        for _ in range(n_generations):
            # evaluate
            for chrom in self.population:
                class_fitness = self._evaluate_fitness_for_class(chrom, X, y, target_class)
                overall_fitness = self._evaluate_overall_fitness(chrom, X, y)
                chrom.fitness = 0.7 * class_fitness + 0.3 * overall_fitness
            
            # next generation
            new_population = []
            sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            new_population.extend([c.copy() for c in sorted_pop[:2]])  # elitism
            
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                new_population.append(child)
            
            self.population = new_population

    def _micro_evolution(self, X, y):
        """
        Extremely lightweight GA update for a tiny batch (often size 1).
        Uses the same class-focused + overall fitness blend.
        """
        target_class = y[0]

        # evaluate
        for chrom in self.population:
            class_fitness = self._evaluate_fitness_for_class(chrom, X, y, target_class)
            overall_fitness = self._evaluate_overall_fitness(chrom, X, y)
            chrom.fitness = 0.7 * class_fitness + 0.3 * overall_fitness

        # one generation: elitism + crossover + mutation
        sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
        new_population = [c.copy() for c in sorted_pop[:2]]

        while len(new_population) < self.population_size:
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            child = self._crossover(p1, p2)
            self._mutate(child)
            new_population.append(child)

        self.population = new_population

    # ---------- Batch update (waves) ----------

    def partial_fit(self, X, y):
        """
        Incremental update with a new batch - enhanced version.
        Handles:
          - class expansion
          - imbalance-aware fitness
          - drift-aware mutation / generations
          - per-class adaptive buffers
        """
        import time
        import numpy as np

        start_time = time.time()

        # ensure classes and update counts
        self._ensure_classes(y)
        self._update_class_counts(y)

        # initialize population if first time
        if len(self.population) == 0:
            self._initialize_population(X, y)

        # update buffers with adaptive sizes
        for c in np.unique(y):
            X_c = X[y == c]
            self.class_buffers[int(c)].extend(X_c.tolist())
            eff_size = self._get_effective_buffer_size(c)
            if len(self.class_buffers[int(c)]) > eff_size:
                self.class_buffers[int(c)] = self.class_buffers[int(c)][-eff_size:]

        # ----------------------------------
        # adaptive cycles & generations
        # ----------------------------------
        total_buffer_size = sum(len(buf) for buf in self.class_buffers.values())
        if total_buffer_size < 200:
            effective_cycles = 1
        elif total_buffer_size < 500:
            effective_cycles = 2
        else:
            effective_cycles = self.n_cycles

        n_active_classes = len([
            c for c in self.seen_classes
            if c in self.class_buffers and len(self.class_buffers[c]) > 10
        ])
        if n_active_classes <= 1:
            effective_generations = max(3, self.generations_per_wave // 2)
        else:
            effective_generations = self.generations_per_wave

        # boost generations under drift
        if self.in_drift_mode:
            effective_generations = int(
                max(1, effective_generations * self.drift_generations_factor)
            )

        classes = sorted(self.seen_classes)

        # ----------------------------------
        # wave training
        # ----------------------------------
        for _ in range(effective_cycles):
            for target_class in classes:
                if (target_class not in self.class_buffers or
                        len(self.class_buffers[target_class]) < 10):
                    continue

                buffer = self.class_buffers[target_class]
                buffer_size = len(buffer)
                n_samples = min(150, buffer_size)

                X_wave = []
                y_wave = []

                # sample target class with recency bias
                indices = self._sample_indices_with_recency_bias(buffer_size, n_samples // 2)
                X_wave.extend([buffer[i] for i in indices])
                y_wave.extend([target_class] * len(indices))

                # other classes
                other_classes = [
                    c for c in classes
                    if c != target_class and c in self.class_buffers and len(self.class_buffers[c]) > 0
                ]
                if other_classes:
                    per_class_other = max(1, (n_samples // 2) // len(other_classes))
                    for c in other_classes:
                        buf_c = self.class_buffers[c]
                        if len(buf_c) == 0:
                            continue
                        n_c = min(per_class_other, len(buf_c))
                        idx_c = self._sample_indices_with_recency_bias(len(buf_c), n_c)
                        X_wave.extend([buf_c[i] for i in idx_c])
                        y_wave.extend([c] * len(idx_c))

                if len(X_wave) > 10:
                    X_wave = np.array(X_wave)
                    y_wave = np.array(y_wave)
                    self._wave_training_fast(X_wave, y_wave, target_class, effective_generations)

        # update best chromosome on this batch
        for chrom in self.population:
            chrom.fitness = self._evaluate_overall_fitness(chrom, X, y)
        self.best_chromosome = max(self.population, key=lambda c: c.fitness)

        # update drift state based on this batch's performance
        self._update_drift_state(X, y)

        self.training_time += time.time() - start_time
        self.n_updates += 1

    def _sample_indices_with_recency_bias(self, buffer_size, n_samples):
        """
        Sample indices from [0, buffer_size) with a mild bias
        towards more recent samples (higher indices).
        """
        import numpy as np
        if buffer_size <= n_samples:
            return np.arange(buffer_size)
        # linear weights: older -> 1.0, newest -> 2.0
        weights = np.linspace(1.0, 2.0, buffer_size)
        weights /= weights.sum()
        return np.random.choice(buffer_size, size=n_samples, replace=False, p=weights)

    # ---------- True per-sample update ----------

    def partial_fit_single(self, x, y):
        """
        Incremental update with a single sample: (x, y).
        Uses an imbalance-aware gate to avoid over-updating on majority classes.
        """
        import time
        import numpy as np

        start_time = time.time()

        # shape to batch
        x = np.asarray(x).reshape(1, -1)
        y = np.asarray([y])

        # class expansion & count update
        self._ensure_classes(y)
        self._update_class_counts(y)

        # init population on first sample
        if len(self.population) == 0:
            self._initialize_population(x, y)
            self.best_chromosome = self.population[0]

        # update buffer for this class with adaptive size
        label = int(y[0])
        self.class_buffers[label].append(x[0].tolist())
        eff_size = self._get_effective_buffer_size(label)
        if len(self.class_buffers[label]) > eff_size:
            self.class_buffers[label] = self.class_buffers[label][-eff_size:]

        # decide whether to perform micro-evolution (avoid hammering majority)
        do_update = True
        total = self._get_total_samples_seen()
        if total > 0:
            count_label = self.class_counts.get(label, 0)
            freq_label = count_label / total
            if freq_label > self.majority_freq_threshold:
                # majority class: only update with some probability
                import numpy as np
                if np.random.rand() > self.majority_update_prob:
                    do_update = False

        if do_update:
            self._micro_evolution(x, y)

        # update best chromosome (using this single sample)
        for chrom in self.population:
            chrom.fitness = self._evaluate_overall_fitness(chrom, x, y)
        self.best_chromosome = max(self.population, key=lambda c: c.fitness)

        # note: we do NOT update drift state here; it's done on batches
        self.training_time += time.time() - start_time
        self.n_updates += 1

    # ===============================
    #  Prediction
    # ===============================

    def predict(self, X):
        """Predict using best chromosome."""
        import numpy as np
        if self.best_chromosome is None:
            return np.zeros(len(X), dtype=int)
        return self.best_chromosome.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        import numpy as np
        if self.best_chromosome is None:
            n_samples = len(X)
            return np.ones((n_samples, max(1, self.n_classes))) / max(1, self.n_classes)
        return self.best_chromosome.predict_proba(X)
