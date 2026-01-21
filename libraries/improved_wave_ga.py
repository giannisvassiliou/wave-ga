"""
Improved Wave-GA with Fixes for Extreme Imbalance
- Rarity-aware fitness
- Protected crossover for rare classes
- Class-specific buffers
- Intensive initial training for new classes
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
        self.class_fitness = {}  # Track fitness per class
        
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
        new.class_fitness = self.class_fitness.copy()
        return new


class ImprovedStreamingWaveGA:
    """
    Improved Wave-Based GA with fixes for extreme imbalance:
    1. Rarity-aware fitness
    2. Protected crossover
    3. Class-specific buffers
    4. Intensive initial training
    """
    
    def __init__(self, n_features, population_size=15, 
                 generations_per_wave=4, n_cycles=2,
                 mutation_rate=0.15, mutation_strength=0.12,
                 buffer_size_per_class=200):
        
        self.n_features = n_features
        self.population_size = population_size
        self.generations_per_wave = generations_per_wave
        self.n_cycles = n_cycles
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.buffer_size_per_class = buffer_size_per_class
        
        self.population = []
        self.best_chromosome = None
        self.seen_classes = set()
        self.n_classes = 0
        
        # Class-specific buffers (FIX 2)
        self.class_buffers = defaultdict(list)
        
        # Track class frequencies for rarity weighting
        self.class_frequencies = {}
        
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
    
    def _calculate_rarity_weight(self, target_class, y_wave):
        """
        FIX 1: Rarity-aware fitness weighting
        Rare classes get higher weight (up to 0.99)
        Common classes get lower weight (down to 0.70)
        """
        class_counts = np.bincount(y_wave, minlength=self.n_classes)
        class_freq = class_counts[target_class] / len(y_wave)
        
        # Inverse frequency weighting
        # 1% class → weight ≈ 0.99
        # 50% class → weight ≈ 0.67
        # 92% class → weight ≈ 0.52
        rarity_weight = 1.0 / (class_freq + 0.01)
        
        # Clip to reasonable range [0.70, 0.99]
        rarity_weight = np.clip(rarity_weight, 0.70, 0.99)
        
        return rarity_weight
    
    def _evaluate_fitness_for_class(self, chromosome, X, y, target_class):
        """One-vs-rest fitness for specific class"""
        y_binary = (y == target_class).astype(int)
        y_pred = chromosome.predict(X)
        y_pred_binary = (y_pred == target_class).astype(int)
        
        # Balanced accuracy for this class
        if len(np.unique(y_binary)) < 2:
            return 0.5
        
        return balanced_accuracy_score(y_binary, y_pred_binary)
    
    def _evaluate_overall_fitness(self, chromosome, X, y):
        """Overall balanced accuracy across all classes"""
        y_pred = chromosome.predict(X)
        return balanced_accuracy_score(y, y_pred)
    
    def _tournament_selection(self, k=3):
        """Select parent via tournament"""
        tournament = np.random.choice(self.population, size=min(k, len(self.population)), replace=False)
        return max(tournament, key=lambda c: c.fitness)
    
    def _crossover_protected(self, parent1, parent2, rare_classes):
        """
        FIX 3: Protected crossover for rare classes
        Rare class weights are copied from better parent (not mixed)
        Common class weights use normal uniform crossover
        """
        child = Chromosome(self.n_classes, self.n_features)
        
        for c in range(self.n_classes):
            if c in rare_classes:
                # Protected: Copy from parent with better fitness for this class
                p1_fitness = parent1.class_fitness.get(c, 0.0)
                p2_fitness = parent2.class_fitness.get(c, 0.0)
                
                if p1_fitness >= p2_fitness:
                    child.weights[c] = parent1.weights[c].copy()
                else:
                    child.weights[c] = parent2.weights[c].copy()
            else:
                # Normal uniform crossover for common classes
                mask = np.random.rand(self.n_features) < 0.5
                child.weights[c] = np.where(mask, parent1.weights[c], parent2.weights[c])
        
        return child
    
    def _mutate(self, chromosome):
        """Gaussian mutation"""
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.randn(self.n_classes, self.n_features) * self.mutation_strength
            chromosome.weights += mutation
    
    def _identify_rare_classes(self, y_wave):
        """
        Identify rare classes (< 10% of samples)
        These get protected crossover
        """
        class_counts = np.bincount(y_wave, minlength=self.n_classes)
        class_freq = class_counts / len(y_wave)
        
        rare_threshold = 0.10  # Classes with < 10% samples are "rare"
        rare_classes = set([c for c in range(self.n_classes) if class_freq[c] < rare_threshold and class_freq[c] > 0])
        
        return rare_classes
    
    def _wave_training(self, X, y, target_class, n_generations, rare_classes):
        """
        Execute wave training for specific class with rarity-aware fitness
        """
        # FIX 1: Calculate rarity weight for this class
        rarity_weight = self._calculate_rarity_weight(target_class, y)
        
        for generation in range(n_generations):
            # Evaluate fitness with rarity-aware weighting
            for chrom in self.population:
                class_fitness = self._evaluate_fitness_for_class(chrom, X, y, target_class)
                overall_fitness = self._evaluate_overall_fitness(chrom, X, y)
                
                # Store class-specific fitness for protected crossover
                chrom.class_fitness[target_class] = class_fitness
                
                # FIX 1: Use rarity-aware weighting
                # Rare classes get weight up to 0.99 (almost pure class focus)
                # Common classes get weight down to 0.70 (balanced)
                chrom.fitness = rarity_weight * class_fitness + (1 - rarity_weight) * overall_fitness
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best 2
            sorted_pop = sorted(self.population, key=lambda c: c.fitness, reverse=True)
            new_population.extend([c.copy() for c in sorted_pop[:2]])
            
            # Generate rest via crossover + mutation
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                # FIX 3: Use protected crossover
                child = self._crossover_protected(parent1, parent2, rare_classes)
                self._mutate(child)
                new_population.append(child)
            
            self.population = new_population
    
    def partial_fit(self, X, y):
        """
        Incremental update with new batch of data
        Includes all fixes for extreme imbalance
        """
        start_time = time.time()
        
        # Detect new classes
        new_classes = set(np.unique(y)) - self.seen_classes
        
        if new_classes:
            # Expand to accommodate new classes
            old_n_classes = self.n_classes
            self.n_classes = len(self.seen_classes | set(np.unique(y)))
            self.seen_classes.update(np.unique(y))
            
            # Expand existing chromosomes
            for chrom in self.population:
                old_weights = chrom.weights
                chrom.weights = np.zeros((self.n_classes, self.n_features))
                chrom.weights[:old_n_classes] = old_weights
                chrom.n_classes = self.n_classes
        else:
            self.seen_classes.update(np.unique(y))
            if self.n_classes == 0:
                self.n_classes = len(self.seen_classes)
        
        # Initialize population if first time
        if len(self.population) == 0:
            self._initialize_population(X, y)
        
        # FIX 2: Update class-specific buffers (per-class quota)
        for c in np.unique(y):
            X_c = X[y == c]
            self.class_buffers[c].extend(X_c.tolist())
            
            # Keep last N samples PER CLASS (not shared across classes)
            if len(self.class_buffers[c]) > self.buffer_size_per_class:
                self.class_buffers[c] = self.class_buffers[c][-self.buffer_size_per_class:]
        
        # Skip training if too few samples
        if len(X) < 10:
            self.training_time += time.time() - start_time
            return
        
        # FIX 4: Intensive initial training for new classes
        if new_classes:
            print(f"    [New classes detected: {new_classes}]")
            for new_class in new_classes:
                if new_class in self.class_buffers and len(self.class_buffers[new_class]) >= 5:
                    # Create training batch for new class
                    X_new_class = np.array(self.class_buffers[new_class])
                    y_new_class = np.array([new_class] * len(X_new_class))
                    
                    # Add some samples from other classes for context
                    other_classes = [c for c in self.seen_classes if c != new_class and c in self.class_buffers]
                    if other_classes:
                        for other_c in other_classes[:3]:  # Max 3 other classes
                            if len(self.class_buffers[other_c]) > 0:
                                other_samples = np.array(self.class_buffers[other_c][:10])
                                X_new_class = np.vstack([X_new_class, other_samples])
                                y_new_class = np.hstack([y_new_class, [other_c] * len(other_samples)])
                    
                    # Intensive training: 3x normal generations
                    rare_classes = self._identify_rare_classes(y_new_class)
                    self._wave_training(
                        X_new_class, y_new_class, new_class,
                        n_generations=self.generations_per_wave * 3,
                        rare_classes=rare_classes
                    )
        
        # Regular wave-based training
        effective_cycles = max(1, self.n_cycles - (self.n_updates // 10))
        effective_generations = self.generations_per_wave
        
        classes = sorted(self.seen_classes)
        
        for cycle in range(effective_cycles):
            for target_class in classes:
                if target_class not in self.class_buffers or len(self.class_buffers[target_class]) < 5:
                    continue
                
                # Sample from buffer with better balancing
                buffer_size = len(self.class_buffers[target_class])
                n_samples = min(150, buffer_size)
                
                X_wave = []
                y_wave = []
                
                # Target class: 50% of samples
                target_n = n_samples // 2
                indices = np.random.choice(buffer_size, min(target_n, buffer_size), replace=False)
                X_wave.extend([self.class_buffers[target_class][i] for i in indices])
                y_wave.extend([target_class] * len(indices))
                
                # Other classes: 50% divided equally
                other_classes = [c for c in classes if c != target_class and c in self.class_buffers]
                if other_classes:
                    n_per_other = (n_samples - len(y_wave)) // len(other_classes)
                    for c in other_classes:
                        if len(self.class_buffers[c]) > 0:
                            n_c = min(n_per_other, len(self.class_buffers[c]))
                            indices = np.random.choice(len(self.class_buffers[c]), n_c, replace=False)
                            X_wave.extend([self.class_buffers[c][i] for i in indices])
                            y_wave.extend([c] * n_c)
                
                if len(X_wave) > 10:
                    X_wave = np.array(X_wave)
                    y_wave = np.array(y_wave)
                    
                    # Identify rare classes for protected crossover
                    rare_classes = self._identify_rare_classes(y_wave)
                    
                    # Wave training with all fixes
                    self._wave_training(X_wave, y_wave, target_class, effective_generations, rare_classes)
        
        # Update best
        for chrom in self.population:
            chrom.fitness = self._evaluate_overall_fitness(chrom, X, y)
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