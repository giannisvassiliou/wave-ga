"""
Continual Learning Baselines for Fair Comparison with Wave-GA
==============================================================

Implements:
1. Experience Replay (ER) - Linear model with replay buffer
2. A-GEM (Averaged Gradient Episodic Memory) - Linear continual learning baseline

Both use the same linear architecture as Wave-GA for fair comparison.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score


class ExperienceReplayLinear:
    """
    Experience Replay baseline for streaming classification.
    
    Features:
    - Same linear multiclass model as Wave-GA (fairness)
    - Per-class replay buffers (same total capacity as Wave-GA)
    - Standard SGD/PA-style updates
    - No wave scheduling (tests if waves add value beyond replay)
    
    Parameters:
    ----------
    n_features : int
        Number of input features
    buffer_size_per_class : int
        Replay buffer size per class (default: 200, matching Wave-GA)
    learning_rate : float
        Learning rate for gradient updates (default: 0.01)
    update_style : str
        'sgd' or 'pa' (Passive-Aggressive) for weight updates
    replay_ratio : float
        Ratio of replay samples to current batch (default: 0.5)
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(
        self, 
        n_features,
        buffer_size_per_class=200,
        learning_rate=0.01,
        update_style='pa',  # 'sgd' or 'pa'
        replay_ratio=0.5,
        random_state=42
    ):
        self.n_features = n_features
        self.buffer_size_per_class = buffer_size_per_class
        self.learning_rate = learning_rate
        self.update_style = update_style
        self.replay_ratio = replay_ratio
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Weight matrix: will expand as classes arrive
        self.W = None
        self.n_classes = 0
        self.classes_seen = set()
        
        # Per-class replay buffers
        self.buffers = {}
        
    def _expand_weight_matrix(self, new_classes):
        """Dynamically expand weight matrix when new classes appear"""
        for c in new_classes:
            if c not in self.classes_seen:
                self.classes_seen.add(c)
                self.n_classes += 1
                
                # Initialize buffer for new class
                self.buffers[c] = {'X': [], 'y': []}
                
                # Expand weight matrix
                if self.W is None:
                    self.W = np.random.randn(1, self.n_features) * 0.01
                else:
                    new_row = np.random.randn(1, self.n_features) * 0.01
                    self.W = np.vstack([self.W, new_row])
    
    def _update_buffers(self, X, y):
        """Update per-class replay buffers with new samples"""
        for c in np.unique(y):
            if c not in self.buffers:
                self.buffers[c] = {'X': [], 'y': []}
            
            # Get samples for this class
            mask = y == c
            X_c = X[mask]
            y_c = y[mask]
            
            # Add to buffer
            self.buffers[c]['X'].extend(X_c)
            self.buffers[c]['y'].extend(y_c)
            
            # Maintain buffer size (sliding window)
            if len(self.buffers[c]['X']) > self.buffer_size_per_class:
                excess = len(self.buffers[c]['X']) - self.buffer_size_per_class
                self.buffers[c]['X'] = self.buffers[c]['X'][excess:]
                self.buffers[c]['y'] = self.buffers[c]['y'][excess:]
    
    def _sample_replay_batch(self, batch_size):
        """Sample uniformly from replay buffers"""
        X_replay = []
        y_replay = []
        
        # Calculate samples per class
        n_classes_with_buffer = sum(1 for c in self.buffers if len(self.buffers[c]['X']) > 0)
        if n_classes_with_buffer == 0:
            return np.array([]), np.array([])
        
        samples_per_class = max(1, batch_size // n_classes_with_buffer)
        
        for c in self.buffers:
            if len(self.buffers[c]['X']) > 0:
                # Sample from this class buffer
                n_samples = min(samples_per_class, len(self.buffers[c]['X']))
                indices = np.random.choice(len(self.buffers[c]['X']), n_samples, replace=False)
                
                X_replay.extend([self.buffers[c]['X'][i] for i in indices])
                y_replay.extend([self.buffers[c]['y'][i] for i in indices])
        
        return np.array(X_replay), np.array(y_replay)
    
    def _compute_scores(self, X):
        """Compute class scores for input X"""
        if self.W is None:
            return np.zeros((len(X), 1))
        return X @ self.W.T  # Shape: (n_samples, n_classes)
    
    def _update_weights_sgd(self, X, y):
        """Standard SGD update"""
        scores = self._compute_scores(X)
        
        for i in range(len(X)):
            y_true = y[i]
            
            # Map label to index
            try:
                y_true_idx = list(self.classes_seen).index(y_true)
            except ValueError:
                continue  # Skip if class not seen yet
            
            # Compute softmax probabilities
            exp_scores = np.exp(scores[i] - np.max(scores[i]))
            probs = exp_scores / np.sum(exp_scores)
            
            # Gradient for cross-entropy loss
            grad = probs.copy()
            grad[y_true_idx] -= 1  # Use index, not raw label
            
            # Update weights
            for c in range(self.n_classes):
                self.W[c] -= self.learning_rate * grad[c] * X[i]
    
    def _update_weights_pa(self, X, y):
        """Passive-Aggressive update (similar to PA-II)"""
        C = 1.0  # Aggressiveness parameter
        
        for i in range(len(X)):
            y_true = y[i]
            scores = self._compute_scores(X[i:i+1])[0]
            
            # Map label to index
            try:
                y_true_idx = list(self.classes_seen).index(y_true)
            except ValueError:
                continue  # Skip if class not seen yet
            
            # Predict
            y_pred_idx = np.argmax(scores)
            
            if y_pred_idx != y_true_idx:
                # Hinge loss
                margin = 1.0
                loss = max(0, margin - (scores[y_true_idx] - scores[y_pred_idx]))
                
                if loss > 0:
                    # PA-II update
                    tau = loss / (np.linalg.norm(X[i])**2 + 1.0 / (2.0 * C))
                    
                    # Update correct class weights (increase)
                    self.W[y_true_idx] += tau * X[i]
                    
                    # Update predicted class weights (decrease)
                    self.W[y_pred_idx] -= tau * X[i]
    
    def partial_fit(self, X, y):
        """
        Incremental training with current batch + replay
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training samples
        y : array-like, shape (n_samples,)
            Target labels
        """
        # Expand weight matrix if new classes appear
        new_classes = set(y) - self.classes_seen
        if new_classes:
            self._expand_weight_matrix(new_classes)
        
        # Update replay buffers with current batch
        self._update_buffers(X, y)
        
        # Sample replay batch
        replay_batch_size = int(len(X) * self.replay_ratio)
        X_replay, y_replay = self._sample_replay_batch(replay_batch_size)
        
        # Combine current batch + replay
        if len(X_replay) > 0:
            X_combined = np.vstack([X, X_replay])
            y_combined = np.concatenate([y, y_replay])
        else:
            X_combined = X
            y_combined = y
        
        # Shuffle combined batch
        indices = np.random.permutation(len(X_combined))
        X_combined = X_combined[indices]
        y_combined = y_combined[indices]
        
        # Update weights
        if self.update_style == 'sgd':
            self._update_weights_sgd(X_combined, y_combined)
        elif self.update_style == 'pa':
            self._update_weights_pa(X_combined, y_combined)
        else:
            raise ValueError(f"Unknown update_style: {self.update_style}")
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            Predicted class labels
        """
        if self.W is None:
            return np.zeros(len(X), dtype=int)
        
        scores = self._compute_scores(X)
        pred_indices = np.argmax(scores, axis=1)
        
        # Convert indices back to actual class labels
        classes_list = list(self.classes_seen)
        return np.array([classes_list[i] for i in pred_indices])


class AGEMLinear:
    """
    A-GEM (Averaged Gradient Episodic Memory) baseline.
    
    A continual learning method that constrains gradient updates to not 
    increase loss on replay buffer (prevents catastrophic forgetting).
    
    Features:
    - Same linear multiclass model as Wave-GA
    - Per-class episodic memory (replay buffers)
    - Gradient projection to prevent forgetting
    - Recognized continual learning baseline
    
    Parameters:
    ----------
    n_features : int
        Number of input features
    buffer_size_per_class : int
        Memory buffer size per class (default: 200)
    learning_rate : float
        Learning rate for gradient descent (default: 0.01)
    memory_strength : float
        Strength of constraint on replay memory (default: 0.5)
    random_state : int
        Random seed
    
    Reference:
    ---------
    Chaudhry et al. "Efficient Lifelong Learning with A-GEM", ICLR 2019
    """
    
    def __init__(
        self,
        n_features,
        buffer_size_per_class=200,
        learning_rate=0.1,  # Increased from 0.01 - SGD needs larger LR
        memory_strength=0.5,
        random_state=42
    ):
        self.n_features = n_features
        self.buffer_size_per_class = buffer_size_per_class
        self.learning_rate = learning_rate
        self.memory_strength = memory_strength
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Weight matrix
        self.W = None
        self.n_classes = 0
        self.classes_seen = set()
        
        # Episodic memory buffers (per-class)
        self.memory = {}
        
    def _expand_weight_matrix(self, new_classes):
        """Expand weight matrix for new classes"""
        for c in new_classes:
            if c not in self.classes_seen:
                self.classes_seen.add(c)
                self.n_classes += 1
                
                # Initialize memory for new class
                self.memory[c] = {'X': [], 'y': []}
                
                # Expand weights
                if self.W is None:
                    self.W = np.random.randn(1, self.n_features) * 0.01
                else:
                    new_row = np.random.randn(1, self.n_features) * 0.01
                    self.W = np.vstack([self.W, new_row])
    
    def _update_memory(self, X, y):
        """Update episodic memory with new samples"""
        for c in np.unique(y):
            if c not in self.memory:
                self.memory[c] = {'X': [], 'y': []}
            
            mask = y == c
            X_c = X[mask]
            y_c = y[mask]
            
            self.memory[c]['X'].extend(X_c)
            self.memory[c]['y'].extend(y_c)
            
            # Maintain buffer size
            if len(self.memory[c]['X']) > self.buffer_size_per_class:
                excess = len(self.memory[c]['X']) - self.buffer_size_per_class
                self.memory[c]['X'] = self.memory[c]['X'][excess:]
                self.memory[c]['y'] = self.memory[c]['y'][excess:]
    
    def _sample_memory_batch(self, batch_size):
        """Sample from episodic memory"""
        X_mem = []
        y_mem = []
        
        n_classes_with_memory = sum(1 for c in self.memory if len(self.memory[c]['X']) > 0)
        if n_classes_with_memory == 0:
            return np.array([]), np.array([])
        
        samples_per_class = max(1, batch_size // n_classes_with_memory)
        
        for c in self.memory:
            if len(self.memory[c]['X']) > 0:
                n_samples = min(samples_per_class, len(self.memory[c]['X']))
                indices = np.random.choice(len(self.memory[c]['X']), n_samples, replace=False)
                
                X_mem.extend([self.memory[c]['X'][i] for i in indices])
                y_mem.extend([self.memory[c]['y'][i] for i in indices])
        
        return np.array(X_mem), np.array(y_mem)
    
    def _compute_scores(self, X):
        """Compute class scores"""
        if self.W is None:
            return np.zeros((len(X), 1))
        return X @ self.W.T
    
    def _compute_gradient(self, X, y):
        """Compute gradient of cross-entropy loss"""
        scores = self._compute_scores(X)
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Gradient
        grad_W = np.zeros_like(self.W)
        
        for i in range(len(X)):
            y_true = y[i]
            # FIX: Map true label to class index
            try:
                y_true_idx = list(self.classes_seen).index(y_true)
            except ValueError:
                # Class not in classes_seen yet (shouldn't happen but be safe)
                continue
            
            grad = probs[i].copy()
            grad[y_true_idx] -= 1  # ← Fixed: use index, not raw label
            
            for c in range(self.n_classes):
                grad_W[c] += grad[c] * X[i]
        
        grad_W /= len(X)
        return grad_W
    
    def _project_gradient(self, g_current, g_memory):
        """
        Project current gradient to not increase loss on memory
        
        This is the core A-GEM mechanism: if gradient would hurt memory,
        project it to be orthogonal to memory gradient.
        """
        # Flatten gradients
        g_current_flat = g_current.flatten()
        g_memory_flat = g_memory.flatten()
        
        # Check if current gradient increases memory loss
        dot_product = np.dot(g_current_flat, g_memory_flat)
        
        if dot_product < 0:
            # Project gradient
            g_memory_norm_sq = np.dot(g_memory_flat, g_memory_flat)
            
            if g_memory_norm_sq > 1e-8:
                projection = (dot_product / g_memory_norm_sq) * g_memory_flat
                g_projected_flat = g_current_flat - projection
                
                # Reshape back
                return g_projected_flat.reshape(g_current.shape)
        
        return g_current
    
    def partial_fit(self, X, y):
        """
        A-GEM training step
        
        1. Compute gradient on current batch
        2. Sample memory batch and compute memory gradient
        3. Project current gradient if it would hurt memory
        4. Update weights with projected gradient
        """
        # Expand for new classes
        new_classes = set(y) - self.classes_seen
        if new_classes:
            self._expand_weight_matrix(new_classes)
        
        # Update episodic memory
        self._update_memory(X, y)
        
        # Compute gradient on current batch
        g_current = self._compute_gradient(X, y)
        
        # Sample memory batch and compute memory gradient
        X_mem, y_mem = self._sample_memory_batch(batch_size=len(X))
        
        if len(X_mem) > 0:
            g_memory = self._compute_gradient(X_mem, y_mem)
            
            # A-GEM: Project gradient to not hurt memory
            g_final = self._project_gradient(g_current, g_memory)
        else:
            # No memory yet, use current gradient
            g_final = g_current
        
        # Update weights
        self.W -= self.learning_rate * g_final
    
    def predict(self, X):
        """Predict class labels"""
        if self.W is None:
            return np.zeros(len(X), dtype=int)
        
        scores = self._compute_scores(X)
        pred_indices = np.argmax(scores, axis=1)
        
        # Convert indices back to actual class labels
        classes_list = list(self.classes_seen)
        return np.array([classes_list[i] for i in pred_indices])


# Wrapper classes for compatibility with existing code structure
class ExperienceReplayWrapper:
    """Wrapper to match the existing interface"""
    def __init__(self, n_features, buffer_size=200, learning_rate=0.01, 
                 update_style='pa', random_state=42):
        self.model = ExperienceReplayLinear(
            n_features=n_features,
            buffer_size_per_class=buffer_size,
            learning_rate=learning_rate,
            update_style=update_style,
            random_state=random_state
        )
    
    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class AGEMWrapper:
    """Wrapper to match the existing interface"""
    def __init__(self, n_features, buffer_size=200, learning_rate=0.1,  # Fixed: 0.01 -> 0.1
                 memory_strength=0.5, random_state=42):
        self.model = AGEMLinear(
            n_features=n_features,
            buffer_size_per_class=buffer_size,
            learning_rate=learning_rate,
            memory_strength=memory_strength,
            random_state=random_state
        )
    
    def partial_fit(self, X, y):
        self.model.partial_fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
