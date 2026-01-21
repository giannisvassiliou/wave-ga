"""
Sophisticated Linear Streaming Classifiers for Proper Baseline Comparison
============================================================================

Implementations of state-of-the-art linear online learning algorithms:
1. Passive-Aggressive (PA, PA-I, PA-II)
2. AROW (Adaptive Regularization of Weight Vectors)
3. Confidence-Weighted (CW) Learning
4. Second-Order Perceptron (SOP)
5. Online Gradient Descent with Adaptive Learning Rates

These are the ACTUAL baselines that should be compared against for linear
streaming classification, not just vanilla SGD.

Author: Based on original papers and implementations
References:
- PA: Crammer et al. (2006) "Online Passive-Aggressive Algorithms"
- AROW: Crammer et al. (2013) "Adaptive Regularization of Weight Vectors"
- CW: Dredze et al. (2008) "Confidence-Weighted Linear Classification"
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, f1_score


# ==============================================================================
# PASSIVE-AGGRESSIVE ALGORITHMS
# ==============================================================================

class PassiveAggressiveClassifier:
    """
    Passive-Aggressive Algorithm for online multiclass classification.
    
    Three variants:
    - PA: Aggressive update (may overfit to noise)
    - PA-I: Soft margin with slack (C parameter)
    - PA-II: Alternative soft margin formulation
    
    Reference: Crammer et al. (2006) "Online Passive-Aggressive Algorithms"
    """
    
    def __init__(self, C=1.0, variant='PA-II', random_state=42):
        """
        Parameters:
        -----------
        C : float, default=1.0
            Aggressiveness parameter. Higher C = more aggressive updates.
            For PA-I and PA-II, this is the slack penalty.
        variant : str, default='PA-II'
            One of 'PA', 'PA-I', 'PA-II'
        random_state : int
            Random seed for initialization
        """
        self.C = C
        self.variant = variant
        self.random_state = random_state
        self.W = None  # Weight matrix [n_classes, n_features]
        self.classes_ = []
        self.n_features = None
        
    def _initialize(self, n_features, n_classes):
        """Initialize weight matrix"""
        np.random.seed(self.random_state)
        self.n_features = n_features
        self.W = np.zeros((n_classes, n_features))
        
    def _get_scores(self, x):
        """Compute scores for all classes"""
        return self.W @ x
    
    def _hinge_loss(self, x, y_true):
        """Compute hinge loss for current example"""
        scores = self._get_scores(x)
        y_true_idx = self.classes_.index(y_true)
        
        # Hinge loss: max(0, 1 - score_correct + max_other_score)
        correct_score = scores[y_true_idx]
        max_wrong_score = np.max(np.delete(scores, y_true_idx))
        
        return max(0, 1 - correct_score + max_wrong_score)
    
    def partial_fit(self, X, y, classes=None):
        """
        Update model with new batch of data.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
        classes : array-like, optional
            All possible class labels
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        # Handle new classes
        if self.W is None:
            if classes is None:
                classes = np.unique(y)
            self.classes_ = list(classes)
            self._initialize(X.shape[1], len(self.classes_))
        else:
            # Expand for new classes
            new_classes = [c for c in np.unique(y) if c not in self.classes_]
            if new_classes:
                old_W = self.W
                self.classes_.extend(new_classes)
                self.W = np.zeros((len(self.classes_), self.n_features))
                self.W[:old_W.shape[0], :] = old_W
        
        # Online update: one example at a time
        for x_i, y_i in zip(X, y):
            self._update(x_i, y_i)
        
        return self
    
    def _update(self, x, y_true):
        """PA update for single example"""
        scores = self._get_scores(x)
        y_true_idx = self.classes_.index(y_true)
        
        # Find prediction
        y_pred_idx = np.argmax(scores)
        
        # If correct, stay passive
        if y_pred_idx == y_true_idx:
            # Check if margin is satisfied
            margin = scores[y_true_idx] - np.max(np.delete(scores, y_true_idx))
            if margin >= 1.0:
                return
        
        # Aggressive update needed
        # Find most confusing wrong class
        wrong_scores = np.copy(scores)
        wrong_scores[y_true_idx] = -np.inf
        y_wrong_idx = np.argmax(wrong_scores)
        
        # Compute loss
        loss = max(0, 1 - scores[y_true_idx] + scores[y_wrong_idx])
        
        if loss == 0:
            return
        
        # Compute update magnitude tau
        x_norm_sq = np.dot(x, x)
        
        if self.variant == 'PA':
            # Hard margin PA
            tau = loss / (2 * x_norm_sq + 1e-10)
        elif self.variant == 'PA-I':
            # Soft margin PA-I
            tau = min(self.C, loss / (2 * x_norm_sq + 1e-10))
        else:  # PA-II
            # Soft margin PA-II
            tau = loss / (2 * x_norm_sq + 1 / (2 * self.C) + 1e-10)
        
        # Update weights
        self.W[y_true_idx] += tau * x
        self.W[y_wrong_idx] -= tau * x
    
    def predict(self, X):
        """Predict class labels"""
        X = np.atleast_2d(X)
        scores = X @ self.W.T
        pred_indices = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in pred_indices])
    
    def score(self, X, y):
        """Return balanced accuracy"""
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)


# ==============================================================================
# AROW (Adaptive Regularization of Weight Vectors)
# ==============================================================================

class AROWClassifier:
    """
    Adaptive Regularization of Weight Vectors for online classification.
    
    Maintains confidence (covariance) for each weight and adapts regularization
    accordingly. More robust to label noise than PA.
    
    Reference: Crammer et al. (2013) "Adaptive Regularization of Weight Vectors"
    """
    
    def __init__(self, r=1.0, random_state=42):
        """
        Parameters:
        -----------
        r : float, default=1.0
            Regularization parameter. Lower r = more conservative updates.
            Typically r ∈ [0.01, 10.0]
        random_state : int
            Random seed
        """
        self.r = r
        self.random_state = random_state
        self.mu = None  # Mean weight vectors [n_classes, n_features]
        self.Sigma = None  # Diagonal covariance [n_classes, n_features]
        self.classes_ = []
        self.n_features = None
        
    def _initialize(self, n_features, n_classes):
        """Initialize mean and covariance"""
        np.random.seed(self.random_state)
        self.n_features = n_features
        self.mu = np.zeros((n_classes, n_features))
        self.Sigma = np.ones((n_classes, n_features))  # Start with unit variance
        
    def partial_fit(self, X, y, classes=None):
        """Update with new batch"""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        # Initialize or expand for new classes
        if self.mu is None:
            if classes is None:
                classes = np.unique(y)
            self.classes_ = list(classes)
            self._initialize(X.shape[1], len(self.classes_))
        else:
            new_classes = [c for c in np.unique(y) if c not in self.classes_]
            if new_classes:
                old_mu = self.mu
                old_Sigma = self.Sigma
                self.classes_.extend(new_classes)
                self.mu = np.zeros((len(self.classes_), self.n_features))
                self.Sigma = np.ones((len(self.classes_), self.n_features))
                self.mu[:old_mu.shape[0], :] = old_mu
                self.Sigma[:old_Sigma.shape[0], :] = old_Sigma
        
        # Update each example
        for x_i, y_i in zip(X, y):
            self._update(x_i, y_i)
        
        return self
    
    def _update(self, x, y_true):
        """AROW update for single example"""
        y_true_idx = self.classes_.index(y_true)
        
        # Compute confidence-weighted scores
        scores = self.mu @ x
        
        # Check if update needed
        y_pred_idx = np.argmax(scores)
        if y_pred_idx == y_true_idx:
            margin = scores[y_true_idx] - np.max(np.delete(scores, y_true_idx))
            if margin >= 1.0:
                return  # Confident and correct
        
        # Find most confusing wrong class
        wrong_scores = np.copy(scores)
        wrong_scores[y_true_idx] = -np.inf
        y_wrong_idx = np.argmax(wrong_scores)
        
        # Compute margin and loss
        m_t = scores[y_true_idx] - scores[y_wrong_idx]
        loss = max(0, 1 - m_t)
        
        if loss == 0:
            return
        
        # Compute confidence (using diagonal covariance)
        # v_t = x^T Σ x for the relevant weight differences
        Sigma_sum = self.Sigma[y_true_idx] + self.Sigma[y_wrong_idx]
        v_t = np.dot(x * x, Sigma_sum)
        
        # Compute update magnitude beta
        beta = 1.0 / (v_t + self.r)
        alpha = loss * beta
        
        # Update mean
        update = alpha * (Sigma_sum * x)
        self.mu[y_true_idx] += update
        self.mu[y_wrong_idx] -= update
        
        # Update covariance (shrink confidence where we updated)
        # Σ_new = Σ - β * Σ * x * x^T * Σ (diagonal approximation)
        covar_update = beta * (Sigma_sum * x * x) * Sigma_sum
        self.Sigma[y_true_idx] -= covar_update
        self.Sigma[y_wrong_idx] -= covar_update
        
        # Ensure covariance stays positive
        self.Sigma = np.maximum(self.Sigma, 1e-8)
    
    def predict(self, X):
        """Predict using mean weights"""
        X = np.atleast_2d(X)
        scores = X @ self.mu.T
        pred_indices = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in pred_indices])
    
    def score(self, X, y):
        """Return balanced accuracy"""
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)


# ==============================================================================
# CONFIDENCE-WEIGHTED LEARNING
# ==============================================================================

class ConfidenceWeightedClassifier:
    """
    Confidence-Weighted Linear Classification.
    
    Maintains Gaussian distribution over weights and updates to maintain
    confidence that new weights will classify correctly.
    
    Reference: Dredze et al. (2008) "Confidence-weighted linear classification"
    """
    
    def __init__(self, eta=0.9, random_state=42):
        """
        Parameters:
        -----------
        eta : float, default=0.9
            Confidence parameter (probability threshold). Higher = more confident.
            Typically eta ∈ [0.5, 0.99]
        """
        self.eta = eta
        self.random_state = random_state
        self.phi = self._inverse_cdf(eta)  # Φ^(-1)(η)
        self.mu = None
        self.Sigma = None
        self.classes_ = []
        self.n_features = None
        
    def _inverse_cdf(self, p):
        """Inverse CDF of standard normal (approximation)"""
        from scipy.stats import norm
        return norm.ppf(p)
    
    def _initialize(self, n_features, n_classes):
        """Initialize"""
        np.random.seed(self.random_state)
        self.n_features = n_features
        self.mu = np.zeros((n_classes, n_features))
        self.Sigma = np.ones((n_classes, n_features))
        
    def partial_fit(self, X, y, classes=None):
        """Update with new batch"""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        if self.mu is None:
            if classes is None:
                classes = np.unique(y)
            self.classes_ = list(classes)
            self._initialize(X.shape[1], len(self.classes_))
        else:
            new_classes = [c for c in np.unique(y) if c not in self.classes_]
            if new_classes:
                old_mu = self.mu
                old_Sigma = self.Sigma
                self.classes_.extend(new_classes)
                self.mu = np.zeros((len(self.classes_), self.n_features))
                self.Sigma = np.ones((len(self.classes_), self.n_features))
                self.mu[:old_mu.shape[0], :] = old_mu
                self.Sigma[:old_Sigma.shape[0], :] = old_Sigma
        
        for x_i, y_i in zip(X, y):
            self._update(x_i, y_i)
        
        return self
    
    def _update(self, x, y_true):
        """CW update"""
        y_true_idx = self.classes_.index(y_true)
        
        scores = self.mu @ x
        y_pred_idx = np.argmax(scores)
        
        # Only update on mistakes
        if y_pred_idx != y_true_idx:
            # Find most confusing wrong class
            wrong_scores = np.copy(scores)
            wrong_scores[y_true_idx] = -np.inf
            y_wrong_idx = np.argmax(wrong_scores)
            
            # Compute confidence-weighted margin
            m_t = scores[y_true_idx] - scores[y_wrong_idx]
            
            # Compute variance
            Sigma_sum = self.Sigma[y_true_idx] + self.Sigma[y_wrong_idx]
            v_t = np.dot(x * x, Sigma_sum)
            
            # Compute update (simplified CW)
            gamma = self.phi * np.sqrt(v_t)
            
            if m_t < gamma:
                # Need update
                alpha = max(0, (gamma - m_t) / v_t)
                
                # Update
                update = alpha * (Sigma_sum * x)
                self.mu[y_true_idx] += update
                self.mu[y_wrong_idx] -= update
                
                # Update covariance (conservative - less shrinkage than AROW)
                beta = alpha * self.phi / np.sqrt(v_t + alpha * self.phi)
                covar_update = beta * (Sigma_sum * x * x) * Sigma_sum
                self.Sigma[y_true_idx] -= 0.5 * covar_update
                self.Sigma[y_wrong_idx] -= 0.5 * covar_update
                
                self.Sigma = np.maximum(self.Sigma, 1e-8)
    
    def predict(self, X):
        """Predict"""
        X = np.atleast_2d(X)
        scores = X @ self.mu.T
        pred_indices = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in pred_indices])
    
    def score(self, X, y):
        """Balanced accuracy"""
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)


# ==============================================================================
# SECOND-ORDER PERCEPTRON
# ==============================================================================

class SecondOrderPerceptron:
    """
    Second-Order Perceptron with diagonal approximation.
    
    Uses second-order information (feature-wise learning rates) for faster
    convergence than standard perceptron.
    
    Similar to diagonal AdaGrad but in online mistake-driven setting.
    """
    
    def __init__(self, a=1.0, random_state=42):
        """
        Parameters:
        -----------
        a : float, default=1.0
            Initial learning rate scaling
        """
        self.a = a
        self.random_state = random_state
        self.W = None
        self.V = None  # Diagonal second-order information
        self.classes_ = []
        self.n_features = None
        
    def _initialize(self, n_features, n_classes):
        """Initialize"""
        np.random.seed(self.random_state)
        self.n_features = n_features
        self.W = np.zeros((n_classes, n_features))
        self.V = np.ones((n_classes, n_features)) * self.a
        
    def partial_fit(self, X, y, classes=None):
        """Update"""
        X = np.atleast_2d(X)
        y = np.atleast_1d(y)
        
        if self.W is None:
            if classes is None:
                classes = np.unique(y)
            self.classes_ = list(classes)
            self._initialize(X.shape[1], len(self.classes_))
        else:
            new_classes = [c for c in np.unique(y) if c not in self.classes_]
            if new_classes:
                old_W = self.W
                old_V = self.V
                self.classes_.extend(new_classes)
                self.W = np.zeros((len(self.classes_), self.n_features))
                self.V = np.ones((len(self.classes_), self.n_features)) * self.a
                self.W[:old_W.shape[0], :] = old_W
                self.V[:old_V.shape[0], :] = old_V
        
        for x_i, y_i in zip(X, y):
            self._update(x_i, y_i)
        
        return self
    
    def _update(self, x, y_true):
        """Second-order perceptron update"""
        y_true_idx = self.classes_.index(y_true)
        
        scores = self.W @ x
        y_pred_idx = np.argmax(scores)
        
        # Update on mistake
        if y_pred_idx != y_true_idx:
            # Find most confusing wrong class
            wrong_scores = np.copy(scores)
            wrong_scores[y_true_idx] = -np.inf
            y_wrong_idx = np.argmax(wrong_scores)
            
            # Compute adaptive learning rate per feature
            # v_t = a + sum of squared updates
            self.V[y_true_idx] += x * x
            self.V[y_wrong_idx] += x * x
            
            # Update with adaptive rate
            lr_true = 1.0 / np.sqrt(self.V[y_true_idx] + 1e-8)
            lr_wrong = 1.0 / np.sqrt(self.V[y_wrong_idx] + 1e-8)
            
            self.W[y_true_idx] += lr_true * x
            self.W[y_wrong_idx] -= lr_wrong * x
    
    def predict(self, X):
        """Predict"""
        X = np.atleast_2d(X)
        scores = X @ self.W.T
        pred_indices = np.argmax(scores, axis=1)
        return np.array([self.classes_[i] for i in pred_indices])
    
    def score(self, X, y):
        """Balanced accuracy"""
        y_pred = self.predict(X)
        return balanced_accuracy_score(y, y_pred)


# ==============================================================================
# WRAPPER FOR INTEGRATION
# ==============================================================================
# linear_streaming_classifiers.py

import numpy as np

class LinearStreamingWrapper:
    def __init__(self, base_classifier, name=None):
        self.model = base_classifier
        self.name = name or type(base_classifier).__name__
        self._is_fitted = False

    def partial_fit(self, X, y):
        """
        Called by your experiment code with numpy arrays X, y.
        After the first call, we consider the model 'fitted'.
        """
        # Most custom online algos will implement their own partial_fit / fit_one
        # If your base_classifier uses another method name, call that here.
        if hasattr(self.model, "partial_fit"):
            self.model.partial_fit(X, y)
        else:
            # Example alternative style, if your algorithms are per-sample:
            for i in range(len(X)):
                self.model.update(X[i], y[i])

        self._is_fitted = True

    def predict(self, X):
        """
        Safe predict: if model has never been trained, just return zeros.
        This avoids 'NoneType has no attribute T' when internal weights are None.
        """
        n = X.shape[0]

        if not self._is_fitted:
            # untrained -> predict majority class 0 (or any default)
            return np.zeros(n, dtype=int)

        # Call the underlying classifier's predict
        if hasattr(self.model, "predict"):
            return self.model.predict(X)

        # If it's per-sample API:
        preds = []
        for i in range(n):
            preds.append(self.model.predict(X[i]))
        return np.array(preds, dtype=int)

class LinearStreamingWrapperold:
    """
    Wrapper to match the interface expected by the evaluation code.
    Provides per-class F1 computation and other utilities.
    """
    
    def __init__(self, base_classifier, name="LinearClassifier"):
        """
        Parameters:
        -----------
        base_classifier : object
            One of PA, AROW, CW, SOP classifiers
        name : str
            Display name
        """
        self.clf = base_classifier
        self.name = name
        self.training_time = 0.0
        
    def partial_fit(self, X, y, classes=None):
        """Update model"""
        import time
        start = time.time()
        self.clf.partial_fit(X, y, classes=classes)
        self.training_time += (time.time() - start)
        return self
    
    def predict(self, X):
        """Predict"""
        return self.clf.predict(X)
    
    def score(self, X, y):
        """Balanced accuracy"""
        return self.clf.score(X, y)
    
    def compute_metrics(self, X_test, y_test):
        """Compute comprehensive metrics"""
        y_pred = self.predict(X_test)
        
        # Balanced accuracy
        ba = balanced_accuracy_score(y_test, y_pred)
        
        # Macro F1
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Per-class F1 (for minority F1)
        unique_classes = np.unique(y_test)
        per_class_f1 = []
        
        for cls in unique_classes:
            y_binary_true = (y_test == cls).astype(int)
            y_binary_pred = (y_pred == cls).astype(int)
            
            if y_binary_true.sum() > 0:  # Only if class exists
                cls_f1 = f1_score(y_binary_true, y_binary_pred, zero_division=0)
                per_class_f1.append(cls_f1)
        
        minf1 = min(per_class_f1) if per_class_f1 else 0.0
        
        return ba, f1, minf1


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_linear_baselines():
    """
    Create all sophisticated linear streaming baselines for comparison.
    
    Returns:
    --------
    dict : {name: wrapped_classifier}
    """
    baselines = {
        'PA (Hard)': LinearStreamingWrapper(
            PassiveAggressiveClassifier(C=1.0, variant='PA'),
            name='PA (Hard)'
        ),
        'PA-I (C=1.0)': LinearStreamingWrapper(
            PassiveAggressiveClassifier(C=1.0, variant='PA-I'),
            name='PA-I (C=1.0)'
        ),
        'PA-II (C=1.0)': LinearStreamingWrapper(
            PassiveAggressiveClassifier(C=1.0, variant='PA-II'),
            name='PA-II (C=1.0)'
        ),
        'PA-II (C=10.0)': LinearStreamingWrapper(
            PassiveAggressiveClassifier(C=10.0, variant='PA-II'),
            name='PA-II (C=10.0)'
        ),
        'AROW (r=1.0)': LinearStreamingWrapper(
            AROWClassifier(r=1.0),
            name='AROW (r=1.0)'
        ),
        'AROW (r=0.1)': LinearStreamingWrapper(
            AROWClassifier(r=0.1),
            name='AROW (r=0.1)'
        ),
        'CW (η=0.9)': LinearStreamingWrapper(
            ConfidenceWeightedClassifier(eta=0.9),
            name='CW (η=0.9)'
        ),
        'SOP': LinearStreamingWrapper(
            SecondOrderPerceptron(a=1.0),
            name='Second-Order Perceptron'
        ),
    }
    
    return baselines


# ==============================================================================
# DEMONSTRATION
# ==============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SOPHISTICATED LINEAR STREAMING CLASSIFIERS")
    print("Proper baselines for streaming classification with:")
    print("  - Sequential class arrival")
    print("  - Concept drift")
    print("  - Class imbalance")
    print("="*80)
    
    # Create simple test stream
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate data with 3 classes
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
    
    # Add class centroids
    for i in range(n_samples):
        X_train[i] += y_train[i] * 2
    
    X_test = np.random.randn(200, n_features)
    y_test = np.random.choice([0, 1, 2], size=200, p=[0.7, 0.2, 0.1])
    for i in range(200):
        X_test[i] += y_test[i] * 2
    
    # Test all classifiers
    baselines = create_linear_baselines()
    
    print("\nTesting on synthetic imbalanced stream (70:20:10)...")
    print(f"{'Classifier':<25} {'BA':<10} {'F1':<10} {'MinF1':<10}")
    print("-"*55)
    
    for name, wrapper in baselines.items():
        # Train in batches
        batch_size = 100
        for i in range(0, n_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            wrapper.partial_fit(X_batch, y_batch)
        
        # Evaluate
        ba, f1, minf1 = wrapper.compute_metrics(X_test, y_test)
        print(f"{name:<25} {ba:>8.4f}  {f1:>8.4f}  {minf1:>8.4f}")
    
    print("\n" + "="*80)
    print("✓ All classifiers tested successfully")
    print("✓ Ready for integration into evaluation framework")
    print("="*80)
