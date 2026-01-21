"""
ENHANCED WAVE-GA vs SOTA + EC BASELINES: Multi-Run Evaluation with Statistical Tests
======================================================================================
Features:
1. All requested models + NEW EC BASELINES (Wave-GA, DE+Waves, StandardDE, WaveIm, ARF, Leveraging Bagging, Hoeffding Tree, ADWIN Bagging, Online Bagging, SGD, PA-II, AROW, CW, SOP)
2. Real-time BA display during training
3. 5 independent runs with different random seeds
4. Statistical significance tests (t-tests) at the end
5. Reports mean ± std for BA, F1, MinF1
6. Comprehensive metrics and runtime comparison

"""
import scikit_posthocs as sp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier
from scipy import stats as scipy_stats
import time
import warnings
warnings.filterwarnings('ignore')
from river import tree, ensemble, forest
import os
from io import StringIO
import requests

import sys
sys.path.append('/mnt/user-data/uploads')
#from streaming_wave_ga_paral_correct import StreamingWaveGA
# from streaming_wave_ga_sample_paral import StandardOnlineGA
# from streaming_wave_ga_sample_paral import perSampleStreamingWaveGA
# from streaming_wave_ga_sample_paral import newperSampleStreamingWaveGA
from ultra_fast_wave_ga_buffer_without_blend import UltraFastWaveGAWithBuffer as wavegaBufferNoBlend
from ultra_fast_wave_ga_nobuffer_with_blend import UltraFastWaveGANoBufferWithBlending as wavegaNoBufferBlend
from ultra_fast_wave_ga_nobuffer_without_blend import UltraFastWaveGANoBuffer as wavegaNoBufferNoBlend
from ultra_fast_wave_ga_nowaves_nobuffer_edit import UltraFastWaveGANoWavesNoBuffer
from ultra_fast_wave_ga_nowaves_edit import UltraFastWaveGANoWaves

from ultra_fast_wave_ga_org250 import UltraFastWaveGA as StreamingWaveGA
#from streaming_wave_ga_sample_paral import StandardOnlineGA

from ultra_fast_wave_ga_nowaves_nobuffer_overall import UltraFastStandardOnlineGA
from improved_wave_ga import ImprovedStreamingWaveGA
from improve_wave_ga_corrected import CorrectedImprovedWaveGA
from linear_streaming_classifiers import (
    PassiveAggressiveClassifier,
    AROWClassifier,
    ConfidenceWeightedClassifier,
    SecondOrderPerceptron,
    LinearStreamingWrapper
)

# Import DE classifiers for EC baseline comparison
#from streaming_de_classifiers import StreamingDE_Waves, StandardDE
from ultra_fast_streaming_de_classifiers_edit3 import UltraFastStreamingDE_Waves, UltraFastStandardDE
# ==============================================================================
# DATA GENERATORS
# ==============================================================================

class ElectricityMarketStream:
    """
    Electricity Market (Elec2) Dataset
    Real-world dataset with concept drift from electricity pricing
    
    Dataset: Australian New South Wales Electricity Market
    - Binary classification (price up vs down)
    - 45,312 instances
    - 8 features
    - Contains concept drift due to changing market conditions
    
    Reference: M. Harries, "Splice-2 Comparative Evaluation: Electricity Pricing", 1999
    """
    
    def __init__(self, samples_per_batch=250, seed=42, data_path=None):
        self.samples_per_batch = samples_per_batch
        self.seed = seed
        self.current_position = 0
        
        # Try multiple sources for the dataset
        df = None
        
        if data_path and os.path.exists(data_path):
            print(f"Loading Elec2 dataset from {data_path}")
            df = pd.read_csv(data_path)
        else:
            # Try downloading from various sources
            sources = [
                # MOA repository
                "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/elec.csv",
                # Alternative source
                "https://www.openml.org/data/get_csv/2419/electricity-normalized.arff"
            ]
            
            for url in sources:
                try:
                    #print(f"Attempting to download Elec2 from: {url}")
                    df = pd.read_csv(url)
                    #print(f"Successfully downloaded Elec2 dataset!")
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
        
        # If all downloads failed, generate synthetic data
        if df is None:
            print("All download attempts failed. Generating synthetic electricity-like data...")
            df = self._generate_synthetic_elec2()
        
        # Prepare features and labels
        # Remove date/time columns if present
        potential_label_cols = ['class', 'target', 'label']
        label_col = None
        for col in potential_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            # Assume last column is the label
            label_col = df.columns[-1]
        
        feature_cols = [col for col in df.columns if col not in [label_col, 'date', 'day', 'time']]
        
        self.X_full = df[feature_cols].values.astype(float)
        
        # Convert labels to numeric binary
        y_raw = df[label_col].values
        if y_raw.dtype == 'object' or isinstance(y_raw[0], str):
            # Convert string labels to binary
            unique_labels = np.unique(y_raw)
            self.y_full = (y_raw == unique_labels[1]).astype(int)
        else:
            self.y_full = y_raw.astype(int)
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_full = scaler.fit_transform(self.X_full)
        
        self.n_samples = len(self.X_full)
        self.n_features = self.X_full.shape[1]
        
        print(f"Elec2 dataset loaded: {self.n_samples} samples, {self.n_features} features")
        print(f"Class distribution: {np.bincount(self.y_full.astype(int))}")
    
    def _generate_synthetic_elec2(self):
        """Generate synthetic electricity market data with concept drift"""
        #np.random.seed(self.seed)
        n_samples = 45000
        n_features = 6
        
        X = []
        y = []
        
        # Simulate changing market conditions
        for i in range(n_samples):
            # Time-dependent features
            t = i / n_samples
            
            # Demand features with seasonal patterns
            demand = 100 + 50 * np.sin(2 * np.pi * t * 365 / n_samples)  # Yearly cycle
            demand += 20 * np.sin(2 * np.pi * t * 7 / n_samples)  # Weekly cycle
            
            # Price features with drift
            drift_factor = 1 + 0.5 * t  # Gradual price increase
            base_price = 50 * drift_factor + np.random.randn() * 10
            
            # Transfer and other features
            transfer = demand * 0.1 + np.random.randn() * 5
            vic_demand = demand * 0.9 + np.random.randn() * 10
            vic_price = base_price * 1.1 + np.random.randn() * 8
            
            features = [
                demand + np.random.randn() * 5,
                base_price + np.random.randn() * 5,
                transfer,
                vic_demand,
                vic_price,
                t  # Time feature
            ]
            
            # Label: price UP if above threshold (with concept drift)
            threshold = 50 + 20 * t  # Threshold drifts over time
            label = 1 if base_price > threshold else 0
            
            X.append(features)
            y.append(label)
        
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['class'] = y
        
        return df
    
    def get_batch(self):
        """Get next batch of samples"""
        if self.current_position >= self.n_samples:
            # Reset to beginning if we've exhausted the dataset
            self.current_position = 0
        
        start_idx = self.current_position
        end_idx = min(start_idx + self.samples_per_batch, self.n_samples)
        
        X_batch = self.X_full[start_idx:end_idx]
        y_batch = self.y_full[start_idx:end_idx]
        
        self.current_position = end_idx
        
        return X_batch, y_batch
    
    def reset(self):
        """Reset to beginning of dataset"""
        self.current_position = 0

class RealWeatherStream:
    """
    Real Weather dataset stream (RainTomorrow prediction)

    Dataset: Australian weather observations (e.g., weatherAUS)
    - ~18k samples
    - Binary target: RainTomorrow (Yes/No)
    - Natural seasonal drift + imbalance
    """

    def __init__(self, samples_per_batch=250, seed=42, data_path=None):
        self.samples_per_batch = samples_per_batch
        self.seed = seed
        self.current_position = 0

        df = None

        # 1) Try user-provided local CSV path first
        if data_path is not None and os.path.exists(data_path):
            print(f"Loading Weather dataset from local file: {data_path}")
            df = pd.read_csv(data_path)
        else:
            # 2) Try downloading from known URLs (may fail with HTTP 500)
            sources = [
                "https://www.openml.org/data/get_csv/5392175/weatherAUS.csv",
            ]
            sources= {}
            for url in sources:
                try:
                    print(f"Downloading Weather dataset from: {url}")
                    df = pd.read_csv(url)
                    print("Weather dataset downloaded successfully!")
                    break
                except Exception as e:
                    print(f"Failed to download Weather dataset from {url}: {e}")
                    df = None

        # 3) Fallback: try some common local filenames
        if df is None:
            fallback_files = [
                "weatherAUS.csv",
                "weather.csv",
                os.path.join("data", "weatherAUS.csv"),
                os.path.join("data", "weather.csv"),
            ]
            for fpath in fallback_files:
                if os.path.exists(fpath):
                    print(f"Found local fallback Weather file: {fpath}")
                    df = pd.read_csv(fpath)
                    break

        # 4) If still nothing, bail with clear message
        if df is None:
            raise RuntimeError(
                "Weather dataset unavailable.\n"
                "Please download the Weather dataset (e.g., weatherAUS.csv) "
                "and pass its path via data_path=... in the experiment args."
            )

        # ------------------------------------------------------------------
        # Preprocessing
        # ------------------------------------------------------------------
        # Drop obvious non-numeric / ID columns if present
        remove_cols = ["Date", "Location", "RISK_MM"]
        df = df.drop(columns=[c for c in remove_cols if c in df.columns],
                     errors='ignore')

        # Target column
        target_col = "RainTomorrow"

        if target_col not in df.columns:
            raise RuntimeError(f"Target column '{target_col}' not found in Weather dataset.")

        # Drop rows with missing target
        df = df.dropna(subset=[target_col])

        # Map target to 0/1
        df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

        # Drop any rows with remaining NaNs
        df = df.dropna()

        # Separate features and keep ONLY numeric ones
        feature_cols = [c for c in df.columns if c != target_col]
        numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_feature_cols) == 0:
            raise RuntimeError("No numeric features found in Weather dataset after filtering.")

        print(f"Using {len(numeric_feature_cols)} numeric features for Weather:")
        print(numeric_feature_cols)

        self.X_full = df[numeric_feature_cols].values.astype(float)
        self.y_full = df[target_col].values.astype(int)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        self.X_full = scaler.fit_transform(self.X_full)

        self.n_samples = len(self.X_full)
        self.n_features = self.X_full.shape[1]

        print(f"Weather dataset loaded: {self.n_samples} samples, {self.n_features} features")
        print(f"Class distribution: {np.bincount(self.y_full)}")

    def get_batch(self):
        """Return next batch of sequential weather data."""
        if self.current_position >= self.n_samples:
            self.current_position = 0

        start = self.current_position
        end = min(start + self.samples_per_batch, self.n_samples)

        X = self.X_full[start:end]
        y = self.y_full[start:end]

        self.current_position = end
        return X, y

    def reset(self):
        """Reset stream to the beginning."""
        self.current_position = 0

class WeatherPredictionStream:
    """
    Weather Prediction with Seasonal Drift
    
    Simulates weather prediction task with:
    - Seasonal patterns (4 seasons)
    - Gradual concept drift
    - Multiple weather types
    - Temperature, humidity, pressure features
    
    Challenge: Seasonal patterns change over time (climate change simulation)
    """
    
    def __init__(self, n_features=20, n_weather_types=4, samples_per_batch=250, seed=43):
        self.n_features = n_features
        self.n_weather_types = n_weather_types  # Sunny, Rainy, Cloudy, Stormy
        self.samples_per_batch = samples_per_batch
        self.seed = seed
        
        np.random.seed(seed)
      

        # Define weather type centroids in feature space
        # Features: temperature, humidity, pressure, wind_speed, etc.
        self.weather_centroids = {
            0: np.array([25, 40, 1015, 5] + [np.random.randn() for _ in range(n_features-4)]),  # Sunny
            1: np.array([15, 80, 1005, 15] + [np.random.randn() for _ in range(n_features-4)]), # Rainy
            2: np.array([18, 60, 1010, 8] + [np.random.randn() for _ in range(n_features-4)]),  # Cloudy
            3: np.array([12, 85, 995, 25] + [np.random.randn() for _ in range(n_features-4)]),  # Stormy
        }
        
        # Seasonal drift patterns
        self.season_modifiers = {
            'spring': np.array([0, 5, 0, 0] + [0]*(n_features-4)),
            'summer': np.array([10, -10, 2, -2] + [0]*(n_features-4)),
            'autumn': np.array([-5, 10, -2, 3] + [0]*(n_features-4)),
            'winter': np.array([-15, 0, -5, 5] + [0]*(n_features-4)),
        }
        
        # Long-term climate drift (gradual warming)
        self.climate_drift = np.array([0.1, 0.05, 0.02, 0.01] + [0.01]*(n_features-4))
        
        self.batch_count = 0
        self.total_samples_generated = 0
    
    def get_season(self, batch_num):
        """Determine current season based on batch number"""
        season_cycle = batch_num % 4
        seasons = ['spring', 'summer', 'autumn', 'winter']
        return seasons[season_cycle]
    
    def get_batch(self):
        """Generate next batch with seasonal drift"""
        self.batch_count += 1
        
        # Apply long-term climate drift
        for weather_type in range(self.n_weather_types):
            self.weather_centroids[weather_type] += self.climate_drift
        
        # Get current season
        current_season = self.get_season(self.batch_count)
        season_modifier = self.season_modifiers[current_season]
        
        X, y = [], []
        
        # Generate samples for each weather type
        samples_per_type = self.samples_per_batch // self.n_weather_types
        
        for weather_type in range(self.n_weather_types):
            base_centroid = self.weather_centroids[weather_type] + season_modifier
            
            for _ in range(samples_per_type):
                # Add noise
                noise_scale = 3.0 + 1.0 * (self.batch_count / 20)  # Increasing uncertainty
                sample = base_centroid + np.random.randn(self.n_features) * noise_scale
                
                X.append(sample)
                y.append(weather_type)
        
        # Shuffle the batch
        indices = np.random.permutation(len(X))
        X = np.array(X)[indices]
        y = np.array(y)[indices]
        
        self.total_samples_generated += len(X)
        
        return X, y

# ==============================================================================
# MODEL WRAPPERS
# ==============================================================================

class RiverHoeffdingTreeWrapper:
    def __init__(self, n_features, seed=42):
        self.model = tree.HoeffdingTreeClassifier(grace_period=50, delta=0.01, tau=0.05)
        self.n_features = n_features
        
    def partial_fit(self, X, y):
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            self.model.learn_one(x_dict, int(y[i]))
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)


class RiverAdaptiveRandomForestWrapper:
    def __init__(self, n_features, seed=42):
        self.model = forest.ARFClassifier(n_models=30, seed=seed, grace_period=150, delta=0.00001)
        self.n_features = n_features
        
    def partial_fit(self, X, y):
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            self.model.learn_one(x_dict, int(y[i]))
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)


class RiverLeveragingBaggingWrapper:
    def __init__(self, n_features, seed=42):
        base_model = tree.HoeffdingTreeClassifier(grace_period=50, delta=0.01, tau=0.05)
        self.model = ensemble.LeveragingBaggingClassifier(model=base_model, n_models=5, seed=seed)
        self.n_features = n_features
        
    def partial_fit(self, X, y):
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            self.model.learn_one(x_dict, int(y[i]))
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)


class RiverADWINBaggingWrapper:
    def __init__(self, n_features, seed=42):
        base_model = tree.HoeffdingTreeClassifier(grace_period=50, delta=0.01, tau=0.05)
        self.model = ensemble.ADWINBaggingClassifier(model=base_model, n_models=5, seed=seed)
        self.n_features = n_features
        
    def partial_fit(self, X, y):
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            self.model.learn_one(x_dict, int(y[i]))
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)


class RiverOnlineBaggingWrapper:
    def __init__(self, n_features, seed=42):
        base_model = tree.HoeffdingTreeClassifier(grace_period=50, delta=0.01, tau=0.05)
        self.model = ensemble.BaggingClassifier(model=base_model, n_models=5, seed=seed)
        self.n_features = n_features
        
    def partial_fit(self, X, y):
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            self.model.learn_one(x_dict, int(y[i]))
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            x_dict = {f'f{j}': float(X[i, j]) for j in range(X.shape[1])}
            pred = self.model.predict_one(x_dict)
            predictions.append(pred if pred is not None else 0)
        return np.array(predictions)


class SimpleOnlineClassifier:
    def __init__(self, n_features, seed=42):
        self.model = SGDClassifier(loss='log_loss', random_state=seed, max_iter=10)
        self.seen_classes = set()
        self.fitted = False
        
    def partial_fit(self, X, y):
        self.seen_classes.update(np.unique(y))
        classes = sorted(list(self.seen_classes))
        try:
            self.model.partial_fit(X, y, classes=classes)
            self.fitted = True
        except:
            pass
    
    def predict(self, X):
        if not self.fitted:
            return np.zeros(len(X), dtype=int)
        try:
            return self.model.predict(X)
        except:
            return np.zeros(len(X), dtype=int)

# ==============================================================================
# MODEL TESTING FUNCTIONS
# ==============================================================================

def test_model(model, stream, n_batches=20, show_progress=False, seed=42):
    """
    Test a single model on a stream and return comprehensive metrics.
    """
    ba_scores = []
    f1_scores = []
    minf1_scores = []  # Per-class minimum F1
    batch_times = []
    
    # Reset stream if it has a reset method
    if hasattr(stream, 'reset'):
        stream.reset()
    
    for batch_idx in range(n_batches):
        X_batch, y_batch = stream.get_batch()
        
        # Ensure we have data
        if len(X_batch) == 0:
            continue
        
        start_time = time.time()
        
        # Prediction
        try:
            y_pred = model.predict(X_batch)
        except:
            # For new classes, some models may fail
            y_pred = np.zeros(len(y_batch))
        
        # Compute metrics
        ba = balanced_accuracy_score(y_batch, y_pred)
        f1_macro = f1_score(y_batch, y_pred, average='macro', zero_division=0)
        
        # Compute per-class F1 and find minimum
        unique_classes = np.unique(y_batch)
        class_f1_scores = []
        for cls in unique_classes:
            cls_mask = (y_batch == cls)
            if np.sum(cls_mask) > 0:
                cls_f1 = f1_score(y_batch == cls, y_pred == cls, zero_division=0)
                class_f1_scores.append(cls_f1)
        
        # minf1 = min(class_f1_scores) if class_f1_scores else 0.0
        #   minority_mask = y_te != 0
        #     if np.sum(minority_mask) > 0:
        #         min_f1 = f1_score(y_te[minority_mask], y_pred[minority_mask], average='macro', zero_division=0)
        #     else:
        #         min_f1 = 0.0
        # Training
        minority_mask = (y_batch != 0)
        
        if np.sum(minority_mask) > 0:
            minf1 = f1_score(
                y_batch[minority_mask],
                y_pred[minority_mask],
                average='macro',
                zero_division=0
            )
        else:
            minf1 = 0.0
        classes = np.unique(np.concatenate([y_batch, np.arange(max(y_batch) + 1)]))
        model.partial_fit(X_batch, y_batch)
        
        elapsed = time.time() - start_time
        
        ba_scores.append(ba)
        f1_scores.append(f1_macro)
        minf1_scores.append(minf1)
        batch_times.append(elapsed)
        if batch_idx+1==n_batches:
            print(np.mean(ba_scores))
        show_progress=True
        if show_progress:
            print(f"  Batch {batch_idx+1:2d}/{n_batches}: BA={ba:.4f}, F1={f1_macro:.4f}, MinF1={minf1:.4f}, Time={elapsed:.4f}s")
    
    return {
        'ba': np.array(ba_scores),
        'f1': np.array(f1_scores),
        'minf1': np.array(minf1_scores),
        'times': np.array(batch_times)
    }


def create_models(n_features, n_classes,seed=42):
    """
    Create all models for comparison.
    """
    np.random.seed(seed)
   
    
    generator=16
 
    models = {
            'StandardGA':UltraFastStandardOnlineGA(n_features=generator,    population_size=70,
        generations_per_wave=10
        ,
        n_cycles=2,

        mutation_rate=0.15,
        mutation_strength=0.12,

        ),

        # EC-based models
        'Wave-GA': StreamingWaveGA(
            n_features=generator,
       
            population_size=70,
        generations_per_wave=10,
        n_cycles=2,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=300,
        alpha=0.5
        ),
    #  }
       'Wave-GA-buffer-noBLend': wavegaBufferNoBlend(
            n_features=generator,
       
            population_size=70,
        generations_per_wave=10,
        n_cycles=2,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=300,
        #alpha=0.5
        ),
        'Wave-GA-nobuffer-bLend': wavegaNoBufferBlend(
            n_features=generator,
       
            population_size=70,
        generations_per_wave=10,
        n_cycles=2,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        #buffer_size=300,
        alpha=0.5
        ),
     'Wave-GA-nobuffer-nobLend': wavegaNoBufferNoBlend(
            n_features=generator,
       
            population_size=70,
        generations_per_wave=10,
        n_cycles=2,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        #buffer_size=300,
        #alpha=0.5
        ),
        'Wave-GA-nowaves':UltraFastWaveGANoWaves(
            n_features=generator,
    
            population_size=70,
        generations_per_wave=10,
        n_cycles=2,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=300,
        alpha=0.5
        ),
             'Wave-GA-nowaves-nobuffer':UltraFastWaveGANoWavesNoBuffer(
             n_features=generator,
    
             population_size=70,
         generations_per_wave=10,
         n_cycles=2,
         mutation_rate=0.15,
         mutation_strength=0.12,
         crossover_type="arithmetic",
    
         alpha=0.5
         ),
    
         'DE+Waves': UltraFastStreamingDE_Waves(
          n_features=generator,
         population_size=70,              # Same as Wave-GA
                  # Same as Wave-GA
         F=0.8,                           # Standard DE mutation factor
         CR=0.9,                          # Standard DE crossover rate
               generations_per_wave=10,          # Same as Wave-GA
         n_cycles=2,     
    
         alpha=0.5                    # Same fitness blending as Wave-GA
    
         ),
    
         'StandardDE': UltraFastStandardDE(
               n_features=generator,
         population_size=70,              # Same as Wave-GA
         generations=24*4,                  # Total: 3 cycles × 8 gen/wave (equiv to Wave-GA)
                 #generations_per_wave=8,          # Same as Wave-GA
         #n_cycles=3,     
    
         F=0.8,                           # Standard DE
         CR=0.9,     
         )
         }
    # PA-II is typically best - always include
    print("SEED "+str(seed))
    models['PA-II (C=1.0)'] = LinearStreamingWrapper(
         PassiveAggressiveClassifier(C=1.0, variant='PA-II', random_state=seed),
         name='PA-II (C=1.0)'
     )
    
    
     # models['PA-II (C=10)'] = LinearStreamingWrapper(
     #     PassiveAggressiveClassifier(C=10.0, variant='PA-II', random_state=seed),
     #     name='PA-II (C=10)'
     # )
    
    
     # AROW - particularly good for noisy/imbalanced data
    models['AROW (r=1.0)'] = LinearStreamingWrapper(
         AROWClassifier(r=1.0, random_state=seed),
         name='AROW (r=1.0)'
     )
    
    
    
    models['AROW (r=0.1)'] = LinearStreamingWrapper(
         AROWClassifier(r=0.1, random_state=seed),
         name='AROW (r=0.1)'
     )
    
     # Confidence-Weighted Learning
    models['CW (η=0.9)'] = LinearStreamingWrapper(
         ConfidenceWeightedClassifier(eta=0.9, random_state=seed),
         name='CW (η=0.9)'
     )
    
    
     # Second-Order Perceptron
    models['SOP'] = LinearStreamingWrapper(
         SecondOrderPerceptron(a=1.0, random_state=seed),
         name='SOP'
     )
    
    
    
     # =========================================================================
     # SIMPLE LINEAR BASELINE (for reference)
     # =========================================================================
     # models['SGD-Online'] = SGDClassifier(
     #     loss='log_loss', max_iter=10, random_state=seed
     # )
    
     # np.random.seed(42)
    models['ARF (SOTA)'] = RiverAdaptiveRandomForestWrapper(n_features=generator, seed=seed)
     #
    
    
    models['Leveraging Bagging'] = RiverLeveragingBaggingWrapper(n_features=generator ,seed=seed)
    
    
    models['Hoeffding Tree'] = RiverHoeffdingTreeWrapper(n_features=generator, seed=seed)
    
    
    models['ADWIN Bagging'] = RiverADWINBaggingWrapper(n_features=generator, seed=seed)
    
    
    
    models['Online Bagging'] = RiverOnlineBaggingWrapper(n_features=generator, seed=seed)
    
    
    
     # models['SGD-Online'] = SimpleOnlineClassifier(n_features=generator, seed=seed)
    
    
     #
     # 'PA-II': PassiveAggressiveClassifier(
     #     C=1.0,
     #     variant='PA-II',
     #     random_state=seed
     # ),
     #
     # 'AROW': AROWClassifier(
     #     r=0.1,
     #     random_state=seed
     # ),
     #
     # 'CW': ConfidenceWeightedClassifier(
     #     eta=0.9,
     #     random_state=seed
     # ),
     #
     # 'SOP': SecondOrderPerceptron(
     #     a=1.0,
     #     random_state=seed
     # ),
    return models


def run_single_experiment(stream_generator, stream_args, exp_name, n_batches=20, seed=42):
    """
    Run a single experiment with one random seed.
    
    Returns:
    --------
    results : dict
        Dictionary mapping model names to their results
    """
    # Create stream
    stream = stream_generator(**stream_args, seed=seed)
    
    # Determine n_features and n_classes
    X_sample, y_sample = stream.get_batch()
    n_features = X_sample.shape[1]
    n_classes = len(np.unique(y_sample))
    
    # Reset stream
    if hasattr(stream, 'reset'):
        stream.reset()
    
    # Create models
    models = create_models(n_features, n_classes, seed)
    
    # Test each model
    results = {}
    
    for model_name, model in models.items():
        print(f"\n  Testing {model_name}...")
        
        # Create fresh stream for this model
        stream = stream_generator(**stream_args, seed=seed)
        
        # Test model
        model_results = test_model(
            model,
            stream,
            n_batches=n_batches,
            show_progress=True,
            seed=seed
        )
        
        results[model_name] = model_results
    
    return results


def run_multirun_experiment(stream_generator, stream_args, exp_name, n_runs=5, n_batches=20):
    """
    Run experiment multiple times with different random seeds and aggregate results.
    
    Returns:
    --------
    all_runs : list of dict
        List of results from each run
    aggregated : dict
        Aggregated statistics across all runs
    """
    all_runs = []
    seeds = [42 + i for i in range(n_runs)]
    
    for run_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_idx}/{n_runs} (seed={seed})")
        print(f"{'='*80}")
        #np.random.seed(42)

        
        run_results = run_single_experiment(
            stream_generator,
            stream_args,
            exp_name,
            n_batches=n_batches,
            seed=seed
        )
        
        all_runs.append(run_results)
    
    # Aggregate results across runs
    aggregated = aggregate_multirun_results(all_runs)
    
    return all_runs, aggregated


def aggregate_multirun_results(all_runs):
    """
    Aggregate results from multiple runs.
    
    Returns:
    --------
    aggregated : dict
        Dictionary with statistics for each model
    """
    model_names = list(all_runs[0].keys())
    aggregated = {}
    
    for model_name in model_names:
        # Collect all BA, F1, MinF1 scores across runs
        ba_all = []
        f1_all = []
        minf1_all = []
        times_all = []
        
        for run in all_runs:
            ba_all.append(run[model_name]['ba'])
            f1_all.append(run[model_name]['f1'])
            minf1_all.append(run[model_name]['minf1'])
            times_all.append(run[model_name]['times'])
        
        # Convert to arrays
        ba_all = np.array(ba_all)  # [n_runs, n_batches]
        f1_all = np.array(f1_all)
        minf1_all = np.array(minf1_all)
        times_all = np.array(times_all)
        avg_ba_values = np.mean(ba_all, axis=1)

        # Compute statistics
        # Final values: average of last 3 batches for each run
        final_ba_values = np.mean(ba_all[:, -3:], axis=1)
        final_f1_values = np.mean(f1_all[:, -3:], axis=1)
        final_minf1_values = np.mean(minf1_all[:, -3:], axis=1)
        
        # Average across batches for each run, then across runs
        avg_ba_per_run = np.mean(ba_all, axis=1)
        avg_f1_per_run = np.mean(f1_all, axis=1)
        avg_minf1_per_run = np.mean(minf1_all, axis=1)
        
        # AUC (area under curve)
        auc_ba_per_run = np.sum(ba_all, axis=1)
        
        # Time statistics
        time_per_batch_per_run = np.mean(times_all, axis=1)
        time_per_batch_mean = np.mean(time_per_batch_per_run)
        time_mean = np.mean(times_all)
        time_std = np.std(times_all)
        
        aggregated[model_name] = {
            # BA statistics
            'avg_ba': np.mean(avg_ba_per_run),
            'avg_ba_std': np.std(avg_ba_per_run),
            'final_ba': np.mean(final_ba_values),
            'final_ba_std': np.std(final_ba_values),
            'auc_ba': np.mean(auc_ba_per_run),
            'auc_ba_std': np.std(auc_ba_per_run),
            
            # F1 statistics
            'avg_f1': np.mean(f1_all),
            'avg_f1_std': np.std(np.mean(f1_all, axis=1)),
            'final_f1': np.mean(np.mean(f1_all[:, -3:], axis=1)),
            'final_f1_std': np.std(np.mean(f1_all[:, -3:], axis=1)),
            
            # Same for MinF1
            'avg_minf1': np.mean(minf1_all),
            'avg_minf1_std': np.std(np.mean(minf1_all, axis=1)),
            'final_minf1': np.mean(np.mean(minf1_all[:, -3:], axis=1)),
            'final_minf1_std': np.std(np.mean(minf1_all[:, -3:], axis=1)),
            
            # Runtime
            'time_per_batch_mean': time_per_batch_mean,
            'time_mean': time_mean,
            'time_std': time_std,
            
            # Raw values for t-tests
            'final_ba_values': final_ba_values,
            'final_f1_values': final_f1_values,
            'final_minf1_values': final_minf1_values,
            'final_avg_ba_values': avg_ba_values,
        }
    
    return aggregated


# ==============================================================================
# REPORTING FUNCTIONS
# ==============================================================================

# ==============================================================================
# REPORTING FUNCTIONS - KEEPING ORIGINAL
# ==============================================================================
# ==============================================================================

def print_aggregate_results(aggregated, exp_name):
    """Print aggregated results with mean ± std"""
    
    print(f"\n{'='*160}")
    print(f"{exp_name} - AGGREGATED RESULTS (Mean ± Std over 5 runs)")
    print(f"{'='*160}")
    print(f"{'Model':<25} {'Avg BA':<18} {'Avg MinF1':<18}{'Final BA':<18} {'Final F1':<18} {'Final MinF1':<18}  {'Time/batch (s)':<18}")
    print(f"{'-'*160}")
    
    # Sort by final BA
    rankings = [(name, data) for name, data in aggregated.items()]
    rankings.sort(key=lambda x: x[1]['avg_ba'], reverse=True)
    
    for rank, (model_name, data) in enumerate(rankings, 1):
        marker = "🏆" if rank == 1 else f"{rank}."
        
        final_ba_str = f"{data['final_ba']:.4f}±{data['final_ba_std']:.4f}"
        final_f1_str = f"{data['final_f1']:.4f}±{data['final_f1_std']:.4f}"
        final_minf1_str = f"{data['final_minf1']:.4f}±{data['final_minf1_std']:.4f}"
        avg_ba_str = f"{data['avg_ba']:.4f}±{data['avg_ba_std']:.4f}"
        auc_ba_str = f"{data['auc_ba']:.3f}±{data['auc_ba_std']:.3f}"
        #time_str = f"{data['time_mean']:.4f}±{data['time_std']:.4f}"
        time_str = f"{float(data['time_mean']):.4f}±{float(data['time_std']):.4f}"

        avg_minf1_str = f"{data['avg_minf1']:.4f}±{data['avg_minf1_std']:.4f}"
        print(f"{marker} {model_name:<22} {avg_ba_str:<18} {avg_minf1_str:<18} {final_ba_str:<18} {final_f1_str:<18} {final_minf1_str:<18}  {time_str:<18}")
    
    print(f"{'-'*160}")


def _cliffs_delta(x, y):
    """
    Cliff's delta effect size: P(x>y) - P(x<y)
    Returns value in [-1, 1]. Positive => x tends to be larger.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    # O(n*m) but n_runs is small (e.g., 30), so fine.
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return (gt - lt) / (len(x) * len(y))


def print_statistical_tests(aggregated, exp_name, alpha=0.05):
    """Print statistical tests: existing t-tests + GECCO-style block at the end."""
    
    print(f"\n{'='*120}")
    print(f"{exp_name} - STATISTICAL SIGNIFICANCE TESTS (t-tests)")
    print(f"{'='*120}")
    print("Comparing last 3 batches performance across runs")
    print(f"{'-'*120}")
    print(f"{'Comparison':<50} {'Metric':<12} {'p-value':<15} {'Significant?':<20}")
    print(f"{'-'*120}")
    
    model_names = list(aggregated.keys())

    # Infer number of runs from stored vectors
    try:
        n_runs = len(aggregated['Wave-GA']['final_ba_values'])
    except Exception:
        n_runs = None

    # # --------------------------
    # # (A) Your existing t-tests
    # # --------------------------
    # for model_name in model_names:
    #     if model_name == 'Wave-GA':
    #         continue
    #
    #     for metric in ['ba', 'f1', 'minf1', 'avg_ba', 'avg_minf1']:
    #         wave_ga_values = aggregated['Wave-GA'][f'final_{metric}_values']
    #         other_values   = aggregated[model_name][f'final_{metric}_values']
    #
    #         if len(wave_ga_values) > 1 and len(other_values) > 1:
    #             t_stat, p_value = scipy_stats.ttest_ind(wave_ga_values, other_values)
    #
    #             if p_value < 0.001:
    #                 sig = "*** (p<0.001)"
    #             elif p_value < 0.01:
    #                 sig = "** (p<0.01)"
    #             elif p_value < 0.05:
    #                 sig = "* (p<0.05)"
    #             else:
    #                 sig = "n.s."
    #
    #             wave_mean = np.mean(wave_ga_values)
    #             other_mean = np.mean(other_values)
    #             direction = ">" if wave_mean > other_mean else "<"
    #
    #             comparison = f"Wave-GA ({wave_mean:.4f}) {direction} {model_name} ({other_mean:.4f})"
    #             metric_name = metric.upper() if metric != 'minf1' else 'MinF1'
    #             print(f"{comparison:<50} {metric_name:<12} {p_value:<15.6f} {sig:<20}")
    #
    # print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    # print(f"{'='*120}")

    # -------------------------------------------------------------------
    # (B) NEW: GECCO-style “Statistical Analysis” block (as requested)
    # -------------------------------------------------------------------
    print("\n" + "="*120)
    print("STATISTICAL ANALYSIS (for GECCO-style reporting)")
    print("="*120)

    # 1) Friedman + (optional) Nemenyi for main comparisons
    # Use per-run aggregate scores (AUC-BA is best; if you only have AUC mean/std,
    # you can switch to final_avg_ba_values or final_ba_values).
    # Here we use AVERAGE BA per run (final_avg_ba_values) because it's already stored.
    metric_for_main = "final_avg_ba_values"

    # Build matrix: shape (n_methods, n_runs)
    main_methods = list(aggregated.keys())
    data_rows = []
    ok = True
    for m in main_methods:
        key = metric_for_main
        if key not in aggregated[m]:
            ok = False
            break
        data_rows.append(np.asarray(aggregated[m][key]))
    if ok and len(data_rows) >= 3:
        try:
            # Friedman test expects repeated measures; here runs play the role of blocks.
            fried_stat, fried_p = scipy_stats.friedmanchisquare(*data_rows)
            print(f"Main comparison: Friedman test on per-run aggregate scores ({metric_for_main})")
            print(f"  Friedman χ² = {fried_stat:.4f}, p = {fried_p:.6g} (α = {alpha})")

            # Optional Nemenyi post-hoc if package is available
            # (If not installed, we still print the intended method.)
            if fried_p < alpha:
                try:
                    import scikit_posthocs as sp
                    import pandas as pd
                    # scikit-posthocs wants columns=methods, rows=blocks (runs)
                    df = pd.DataFrame({m: aggregated[m][metric_for_main] for m in main_methods})
                    nemenyi = sp.posthoc_nemenyi_friedman(df.values)
                    nemenyi.index = main_methods
                    nemenyi.columns = main_methods
                    print("  Nemenyi post-hoc p-value matrix (rows/cols = methods):")
                    # Print compactly
                    with np.printoptions(precision=4, suppress=True):
                        print(nemenyi.values)
                except Exception as e:
                    print("  Nemenyi post-hoc: (not computed here — install scikit-posthocs to enable)")
        except Exception as e:
            print("Main comparison: Friedman test failed to compute (check stored per-run vectors).")
    else:
        print("Main comparison: Friedman + Nemenyi not computed (need ≥3 methods and per-run vectors).")

    # 2) Wilcoxon + Cliff’s delta for ablations (Wave-GA vs its variants)
    print("\nAblations: Wilcoxon signed-rank + Cliff’s delta (paired per-run aggregates)")
    ablation_candidates = [m for m in aggregated.keys() if (m.lower().startswith("wave-ga") or m.lower().startswith("standardg")) and m != "Wave-GA"]
    if len(ablation_candidates) == 0:
        print("  No Wave-GA ablation variants detected (expected names like 'Wave-GA-nowaves', etc.).")
    else:
        x = np.asarray(aggregated["Wave-GA"][metric_for_main])
        for m in ablation_candidates:
            y = np.asarray(aggregated[m][metric_for_main])
            # paired Wilcoxon: requires same length and pairing across seeds
            if len(x) == len(y) and len(x) > 1:
                try:
                    w_stat, w_p = scipy_stats.wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
                    delta = _cliffs_delta(x, y)
                    sig = "SIGNIFICANT" if w_p < alpha else "n.s."
                    print(f"  Wave-GA vs {m}: Wilcoxon p={w_p:.6g} ({sig}, α={alpha}); Cliff’s δ={delta:.3f}")
                except Exception:
                    print(f"  Wave-GA vs {m}: Wilcoxon failed (check data, ties, or zero-differences).")
            else:
                print(f"  Wave-GA vs {m}: skipped (requires equal-length paired per-run vectors).")

    # 3) Print the exact requested “output” lines (verbatim)
    print("\nRequested reporting checklist (printed for paper write-up):")
    print('  - Add one subsection: "Statistical Analysis"')
    print("  - Use Friedman + Nemenyi for all main comparisons")
    print("  - Use Wilcoxon + Cliff’s delta for ablations")
    print(f"  - Set α = {alpha}")
    if n_runs is not None:
        print(f"  - State number of runs (e.g., {n_runs})")
    else:
        print("  - State number of runs (e.g., 30)")
    print("="*120)

    """Print statistical significance tests comparing all methods"""
    
    print(f"\n{'='*120}")
    print(f"{exp_name} - STATISTICAL SIGNIFICANCE TESTS (t-tests)")
    print(f"{'='*120}")
    print("Comparing last 3 batches performance across 5 runs")
    print(f"{'-'*120}")
    print(f"{'Comparison':<50} {'Metric':<12} {'p-value':<15} {'Significant?':<20}")
    print(f"{'-'*120}")
    
    model_names = list(aggregated.keys())
    
    # Compare Wave-GA vs all others
    for model_name in model_names:
        if model_name == 'Wave-GA':
            continue
        
        # Test BA, F1, and MinF1
        for metric in ['ba', 'f1', 'minf1']:
            wave_ga_values = aggregated['Wave-GA'][f'final_{metric}_values']
            other_values = aggregated[model_name][f'final_{metric}_values']
            
            if len(wave_ga_values) > 1 and len(other_values) > 1:
                # Perform two-tailed t-test
                t_stat, p_value = scipy_stats.ttest_ind(wave_ga_values, other_values)
                
                # Determine significance
                if p_value < 0.001:
                    sig = "*** (p<0.001)"
                elif p_value < 0.01:
                    sig = "** (p<0.01)"
                elif p_value < 0.05:
                    sig = "* (p<0.05)"
                else:
                    sig = "n.s."
                
                # Determine direction
                wave_mean = np.mean(wave_ga_values)
                other_mean = np.mean(other_values)
                direction = ">" if wave_mean > other_mean else "<"
                
                comparison = f"Wave-GA ({wave_mean:.4f}) {direction} {model_name} ({other_mean:.4f})"
                metric_name = metric.upper() if metric != 'minf1' else 'MinF1'
                print(f"{comparison:<50} {metric_name:<12} {p_value:<15.6f} {sig:<20}")
    
    print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print(f"{'='*120}")

    """Print statistical significance tests comparing all methods"""
    
    print(f"\n{'='*120}")
    print(f"{exp_name} - STATISTICAL SIGNIFICANCE TESTS (t-tests)")
    print(f"{'='*120}")
    print("Comparing last 3 batches performance across 5 runs")
    print(f"{'-'*120}")
    print(f"{'Comparison':<50} {'Metric':<12} {'p-value':<15} {'Significant?':<20}")
    print(f"{'-'*120}")
    
    # model_names = list(aggregated.keys())
    #
    # # Compare Wave-GA vs all others
    # for model_name in model_names:
    #     if model_name == 'Wave-GA':
    #         continue
    #
    #     # Test BA, F1, and MinF1
    #     for metric in ['ba', 'f1', 'minf1']:
    #         wave_ga_values = aggregated['Wave-GA'][f'final_{metric}_values']
    #         other_values = aggregated[model_name][f'final_{metric}_values']
    #
    #         if len(wave_ga_values) > 1 and len(other_values) > 1:
    #             # Perform two-tailed t-test
    #             t_stat, p_value = scipy_stats.ttest_ind(wave_ga_values, other_values)
    #
    #             # Determine significance
    #             if p_value < 0.001:
    #                 sig = "*** (p<0.001)"
    #             elif p_value < 0.01:
    #                 sig = "** (p<0.01)"
    #             elif p_value < 0.05:
    #                 sig = "* (p<0.05)"
    #             else:
    #                 sig = "n.s."
    #
    #             # Determine direction
    #             wave_mean = np.mean(wave_ga_values)
    #             other_mean = np.mean(other_values)
    #             direction = ">" if wave_mean > other_mean else "<"
    #
    #             comparison = f"Wave-GA ({wave_mean:.4f}) {direction} {model_name} ({other_mean:.4f})"
    #             metric_name = metric.upper() if metric != 'minf1' else 'MinF1'
    #             print(f"{comparison:<50} {metric_name:<12} {p_value:<15.6f} {sig:<20}")
    #
    # print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    # print(f"{'='*120}")


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("ENHANCED WAVE-GA + EC BASELINES: Multi-Run Evaluation")
    print("Real-time BA display + Statistical significance tests")
    print("Running NEW EXPERIMENTS: Electricity Market + Weather Prediction")
    print("█"*80)
    
    # Store all results
    all_experiment_results = {}
    
    # Define the 2 experiments
    experiments = [
        # {
        #     'name': 'Electricity Market (Elec2)',
        #     'generator': ElectricityMarketStream,
        #     'args': {'samples_per_batch': 250},
        #     'n_batches': 100  # 45,000 samples / 250 per batch
        # },
             {
            'name': 'Weather (Real Dataset)',
            'generator': RealWeatherStream,
            'args': {'samples_per_batch': 250},
                    'data_path': r'weatherAUS.csv',

            'n_batches':72 
    #72   # 18,000 samples / 250 per batch
        }

    ]
    
    # Run each experiment
    for exp_idx, exp in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {exp_idx}/2: {exp['name']}")
        print(f"{'#'*80}")
        
        all_runs, aggregated = run_multirun_experiment(
            exp['generator'],
            exp['args'],
            exp['name'],
            n_runs=15,
            n_batches=exp['n_batches']
        )
        
        # Store results
        all_experiment_results[exp['name']] = {
            'all_runs': all_runs,
            'aggregated': aggregated
        }
        
        # Print results
        print_aggregate_results(aggregated, exp['name'])
        
        # Print statistical tests
        print_statistical_tests(aggregated, exp['name'])
    
    # Final summary across all experiments
    print("\n" + "="*100)
    print("FINAL SUMMARY: Wave-GA Performance Across All Experiments")
    print("="*100)
    print(f"{'Experiment':<45} {'Wave-GA Final BA':<20} {'Best Baseline':<20} {'Best BA':<15}")
    print("-"*100)
    
    for exp in experiments:
        exp_name = exp['name']
        agg = all_experiment_results[exp_name]['aggregated']
        
        wave_ga_ba = agg['Wave-GA']['final_ba']
        wave_ga_std = agg['Wave-GA']['final_ba_std']
        
        # Find best baseline
        best_ba = 0.0
        best_name = ""
        for model_name, data in agg.items():
            if model_name != 'Wave-GA':
                if data['final_ba'] > best_ba:
                    best_ba = data['final_ba']
                    best_name = model_name
        
        wave_str = f"{wave_ga_ba:.4f}±{wave_ga_std:.4f}"
        best_str = f"{best_ba:.4f}"
        
        print(f"{exp_name:<45} {wave_str:<20} {best_name:<20} {best_str:<15}")
    
    print("="*100)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("="*80)
