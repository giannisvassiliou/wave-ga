"""
ENHANCED WAVE-GA vs SOTA + EC BASELINES: Multi-Run Evaluation with Statistical Tests
======================================================================================

Features:
1. All requested models + NEW EC BASELINES (Wave-GA, DE+Waves, StandardDE, WaveIm, ARF, Leveraging Bagging, Hoeffding Tree, ADWIN Bagging, Online Bagging, SGD, PA-II, AROW, CW, SOP)
2. Real-time BA display during training
3. 30 independent runs with different random seeds
4. Statistical significance tests (t-tests) at the end
5. Reports mean ± std for BA, F1, MinF1
6. Comprehensive metrics and runtime comparison

DATASET STRATEGY:
- Phase 1: First 100K samples (200 batches × 500 samples) - comprehensive evaluation
- Phase 2: First 250K samples (500 batches × 500 samples) - scalability test (optional)
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
from ultra_fast_wave_ga_nowaves_nobuffer_edit import UltraFastWaveGANoWavesNoBuffer
from ultra_fast_wave_ga_nowaves_edit import UltraFastWaveGANoWaves
from ultra_fast_wave_ga_org250 import UltraFastWaveGA as StreamingWaveGA
from ultra_fast_wave_ga_buffer_without_blend import UltraFastWaveGAWithBuffer as wavegaBufferNoBlend
from ultra_fast_wave_ga_nobuffer_with_blend import UltraFastWaveGANoBufferWithBlending as wavegaNoBufferBlend
from ultra_fast_wave_ga_nobuffer_without_blend import UltraFastWaveGANoBuffer as wavegaNoBufferNoBlend
#from ultra_fast_wave_ga import UltraFastWaveGA as StreamingWaveGA
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

class AirlinesStream:
    """
    Airlines Dataset - Flight Delay Prediction
    Real-world dataset with recurring seasonal drift
    
    Dataset: Flight delay prediction
    - Binary classification (delayed yes/no)
    - 539,383 instances (using first 100K for main eval, 250K for scalability)
    - 7 features
    - Contains recurring seasonal drift patterns
    
    Source: https://www.openml.org/d/1169
    Reference: Ikonomovska et al., "Regression Trees from Data Streams with Drift Detection", 2009
    """
    
    def __init__(self, samples_per_batch=500, seed=42, data_path=None, max_samples=100000):
        """
        Initialize Airlines stream
        
        Args:
            samples_per_batch: Batch size (default 500, larger than synthetic experiments)
            seed: Random seed
            data_path: Path to airlines.csv file
            max_samples: Maximum samples to load (100000 for Phase 1, 250000 for Phase 2)
        """
        self.samples_per_batch = samples_per_batch
        self.seed = seed
        self.current_position = 0
        self.max_samples = max_samples
        
        # Try multiple sources for the dataset
        df = None
        
        if data_path and os.path.exists(data_path):
            print(f"Loading Airlines dataset from {data_path}")
            df = pd.read_csv(data_path, nrows=max_samples)
        else:
            # Try downloading from OpenML
            sources = [
                # OpenML direct link
                "https://www.openml.org/data/get_csv/1169/phpvcoG8S",
                # Alternative source
                "https://raw.githubusercontent.com/scikit-multiflow/streaming-datasets/master/airlines.csv"
            ]
            
            for url in sources:
                try:
                    print(f"Attempting to download Airlines from: {url}")
                    df = pd.read_csv(url, nrows=max_samples)
                    print(f"Successfully downloaded Airlines dataset!")
                    break
                except Exception as e:
                    print(f"Failed to download from {url}: {e}")
                    continue
        
        # If all downloads failed, generate synthetic data
        if df is None:
            print("All download attempts failed. Generating synthetic airlines-like data...")
            df = self._generate_synthetic_airlines()
        
        # Prepare features and labels
        # Airlines dataset structure:
        # Columns: Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length, Delay (target)
        
        potential_label_cols = ['Delay', 'delay', 'class', 'target', 'label']
        label_col = None
        for col in potential_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            # Assume last column is the label
            label_col = df.columns[-1]
            print(f"Warning: No standard label column found, using last column: {label_col}")
        
        # Feature engineering for categorical columns
        feature_cols = [col for col in df.columns if col != label_col]
        
        # Handle categorical columns
        df_processed = df.copy()
        categorical_cols = df_processed[feature_cols].select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            print(f"Encoding categorical columns: {list(categorical_cols)}")
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_cols:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        
        self.X_full = df_processed[feature_cols].values.astype(float)
        
        # Convert labels to binary (0 = no delay, 1 = delay)
        y_raw = df_processed[label_col].values
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
        
        print(f"Airlines dataset loaded: {self.n_samples} samples, {self.n_features} features")
        print(f"Batch size: {samples_per_batch} samples")
        print(f"Total batches: {self.n_samples // samples_per_batch}")
        print(f"Class distribution: {np.bincount(self.y_full.astype(int))}")
        class_ratio = np.bincount(self.y_full.astype(int))
        if len(class_ratio) == 2:
            print(f"Class balance: {class_ratio[0]/sum(class_ratio)*100:.1f}% / {class_ratio[1]/sum(class_ratio)*100:.1f}%")
    
    def _generate_synthetic_airlines(self):
        """Generate synthetic airlines data with seasonal patterns"""
        np.random.seed(self.seed)
        n_samples = self.max_samples
        
        X = []
        y = []
        
        # Simulate seasonal flight delay patterns
        for i in range(n_samples):
            # Time features
            t = i / n_samples
            
            # Seasonal component (4 seasons)
            season = int((i % 1000) / 250)  # 0=winter, 1=spring, 2=summer, 3=fall
            
            # Airline (encoded 0-10)
            airline = np.random.randint(0, 10)
            
            # Flight number (encoded)
            flight = np.random.randint(100, 999)
            
            # Airports (encoded 0-50)
            airport_from = np.random.randint(0, 50)
            airport_to = np.random.randint(0, 50)
            
            # Day of week (0-6)
            day_of_week = i % 7
            
            # Time of day (0-24)
            time_of_day = np.random.randint(0, 24)
            
            # Flight length (in hours)
            flight_length = 1 + 10 * np.random.rand()
            
            # Delay probability depends on multiple factors
            delay_prob = 0.3  # base probability
            
            # Seasonal effects (winter = more delays)
            if season == 0:  # winter
                delay_prob += 0.15
            elif season == 2:  # summer
                delay_prob += 0.10  # also high travel season
            
            # Time of day effects (rush hours = more delays)
            if time_of_day in [7, 8, 17, 18]:
                delay_prob += 0.10
            
            # Weekend effects
            if day_of_week in [5, 6]:
                delay_prob -= 0.05
            
            # Length effects (longer flights = more delays)
            delay_prob += (flight_length / 20)
            
            # Add some drift over time
            drift = 0.1 * np.sin(2 * np.pi * t * 4)  # 4 cycles
            delay_prob += drift
            
            # Clip probability
            delay_prob = np.clip(delay_prob, 0, 1)
            
            features = [
                airline,
                flight % 100,  # Simplify
                airport_from,
                airport_to,
                day_of_week,
                time_of_day,
                flight_length
            ]
            
            # Generate delay label
            label = 1 if np.random.rand() < delay_prob else 0
            
            X.append(features)
            y.append(label)
        
        df = pd.DataFrame(X, columns=['Airline', 'Flight', 'AirportFrom', 'AirportTo', 
                                      'DayOfWeek', 'Time', 'Length'])
        df['Delay'] = y
        
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
        """Reset stream to beginning"""
        self.current_position = 0


class WeatherPredictionStream:
    """
    Weather Prediction with Seasonal Drift
    Synthetic dataset with predictable seasonal patterns
    
    Features:
    - Multiclass classification (weather types)
    - Seasonal patterns with drift
    - Controlled concept drift for comparison
    """
    
    def __init__(self, n_features=20, n_weather_types=4, samples_per_batch=250, seed=42):
        self.n_features = n_features
        self.n_weather_types = n_weather_types
        self.samples_per_batch = samples_per_batch
        self.seed = seed
        self.batch_count = 0
        
        # Season definitions
        self.seasons = ['winter', 'spring', 'summer', 'fall']
        self.season_length = 10  # batches per season
        
        print(f"Weather dataset: {n_features} features, {n_weather_types} weather types")
        print(f"Seasonal cycle: {len(self.seasons)} seasons × {self.season_length} batches")
    
    def _get_current_season(self):
        """Get current season based on batch count"""
        season_idx = (self.batch_count // self.season_length) % len(self.seasons)
        return self.seasons[season_idx]
    
    def get_batch(self):
        """Generate a batch with seasonal patterns"""
        np.random.seed(self.seed + self.batch_count)
        
        season = self._get_current_season()
        X = []
        y = []
        
        for _ in range(self.samples_per_batch):
            # Generate features
            features = np.random.randn(self.n_features)
            
            # Seasonal modifications
            if season == 'winter':
                # Cold, low humidity
                features[0] -= 2.0  # temperature
                features[1] -= 1.0  # humidity
                # More likely: snow, cold rain
                weather_probs = [0.4, 0.3, 0.2, 0.1]
            elif season == 'spring':
                # Moderate temp, high humidity
                features[0] += 0.5
                features[1] += 1.5
                # More likely: rain, clouds
                weather_probs = [0.2, 0.4, 0.3, 0.1]
            elif season == 'summer':
                # Hot, variable humidity
                features[0] += 2.0
                # More likely: sun, some rain
                weather_probs = [0.1, 0.2, 0.1, 0.6]
            else:  # fall
                # Cooling, moderate
                features[0] -= 0.5
                features[1] += 0.5
                # Balanced
                weather_probs = [0.25, 0.25, 0.25, 0.25]
            
            # Select weather type based on seasonal probabilities
            weather_type = np.random.choice(self.n_weather_types, p=weather_probs)
            
            # Add seasonal signal to features
            features[2] = {'winter': 0, 'spring': 1, 'summer': 2, 'fall': 3}[season]
            
            X.append(features)
            y.append(weather_type)
        
        self.batch_count += 1
        
        return np.array(X), np.array(y)
    
    def reset(self):
        """Reset to beginning"""
        self.batch_count = 0

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
# MODEL INITIALIZATION - KEEPING ALL ORIGINAL SIGNATURES
# ==============================================================================

def initialize_models(n_features, n_classes,seeda):
    """
    Initialize all models with consistent signatures
    IMPORTANT: This function is kept exactly as is - DO NOT MODIFY
    
    """
    
    seed=seeda
    np.random.seed(seed)
   
    
    generator=7

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


# ==============================================================================
# EVALUATION FUNCTIONS - KEEPING ORIGINAL LOGIC
# ==============================================================================

def run_single_experiment(stream_generator, generator_args, model_dict, n_batches, exp_name, run_idx):
    """
    Run a single experiment with one random seed
    KEEPING ORIGINAL FUNCTION SIGNATURE AND LOGIC
    """
    
    print(f"\n{'='*80}")
    print(f"Run {run_idx + 1} - {exp_name}")
    print(f"{'='*80}")
    
    # Initialize stream
    stream = stream_generator(**generator_args)
    
    # Get first batch to determine dimensions
    X_sample, y_sample = stream.get_batch()
    stream.reset()
    
    n_features = X_sample.shape[1]
    n_classes = len(np.unique(y_sample))
    
    print(f"Stream initialized: {n_features} features, {n_classes} classes")
    
    # Initialize models
    models = model_dict if model_dict is not None else initialize_models(n_features, n_classes,generator_args['seed'])
    
    # Storage for metrics
    results = {
        name: {
            'ba_scores': [],
            'f1_scores': [],
            'minf1_scores': [],
            'time_per_batch': []
        }
        for name in models.keys()
    }
    
    # Prequential evaluation
    for batch_idx in range(n_batches):
        X_batch, y_batch = stream.get_batch()
        
        # Test then Train for each model
        for model_name, model in models.items():
            try:
                start_time = time.time()
                
                # TEST: Predict on current batch
                if hasattr(model, 'predict'):
                    y_pred = model.predict(X_batch)
                elif hasattr(model, 'predict_many'):
                    y_pred = []
                    for x in X_batch:
                        pred = model.predict_one({i: x[i] for i in range(len(x))})
                        if pred is not None:
                            y_pred.append(pred)
                        else:
                            y_pred.append(0)
                    y_pred = np.array(y_pred)
                else:
                    y_pred = np.zeros(len(y_batch))
                
                # Calculate metrics
                if len(y_pred) == len(y_batch) and len(np.unique(y_batch)) > 1:
                    ba = balanced_accuracy_score(y_batch, y_pred)
                    
                    # F1 scores
                    f1_macro = f1_score(y_batch, y_pred, average='macro', zero_division=0)
                    
                    # MinF1: minimum per-class F1
                    f1_per_class = f1_score(y_batch, y_pred, average=None, zero_division=0)
                    min_f1 = np.min(f1_per_class) if len(f1_per_class) > 0 else 0.0
                else:
                    ba = 0.0
                    f1_macro = 0.0
                    min_f1 = 0.0
                
                # TRAIN: Update model
                if hasattr(model, 'partial_fit'):
                    classes = np.unique(np.concatenate([y_batch, np.arange(n_classes)]))
                    model.partial_fit(X_batch, y_batch)
                elif hasattr(model, 'learn_many'):
                    X_dict = [
                        {i: X_batch[j, i] for i in range(X_batch.shape[1])}
                        for j in range(len(X_batch))
                    ]
                    model.learn_many(X_dict, y_batch)
                
                elapsed = time.time() - start_time
                
                # Store results
                results[model_name]['ba_scores'].append(ba)
                results[model_name]['f1_scores'].append(f1_macro)
                results[model_name]['minf1_scores'].append(min_f1)
                results[model_name]['time_per_batch'].append(elapsed)
                
            except Exception as e:
                print(f"Error with {model_name} at batch {batch_idx}: {e}")
                results[model_name]['ba_scores'].append(0.0)
                results[model_name]['f1_scores'].append(0.0)
                results[model_name]['minf1_scores'].append(0.0)
                results[model_name]['time_per_batch'].append(0.0)
        
        # Display progress every 20 batches
        print(batch_idx,end=" ")
        # if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
        
        if True:
            print(f"\n--- Batch {batch_idx + 1}/{n_batches} ---")
            print(f"{'Model':<25} {'BA':<10} {'F1':<10} {'MinF1':<10} {'Time(s)':<10}")
            print("-" * 65)
            
            for model_name in sorted(results.keys()):
                if len(results[model_name]['ba_scores']) > 0:
                    current_ba = results[model_name]['ba_scores'][-1]
                    current_f1 = results[model_name]['f1_scores'][-1]
                    current_minf1 = results[model_name]['minf1_scores'][-1]
                    current_time = results[model_name]['time_per_batch'][-1]
                    
                    print(f"{model_name:<25} {current_ba:<10.4f} {current_f1:<10.4f} {current_minf1:<10.4f} {current_time:<10.4f}")
    
    return results


def run_multirun_experiment(stream_generator, generator_args, exp_name, n_runs=1, n_batches=100):
    """
    Run multiple independent experiments and aggregate results
    KEEPING ORIGINAL FUNCTION SIGNATURE
    """
    
    all_runs = []
    
    for run_idx in range(n_runs):
        # Update seed for each run
        run_args = generator_args.copy()
        run_args['seed'] = 42 + run_idx
        
        # Run experiment
        results = run_single_experiment(
            stream_generator,
            run_args,
            None,
            n_batches,
            exp_name,
            run_idx
        )
        
        all_runs.append(results)
    
    # Aggregate results
    aggregated = {}
    model_names = all_runs[0].keys()
    
    for model_name in model_names:
        # Collect all BA, F1, MinF1 scores across runs
        ba_all = []
        f1_all = []
        minf1_all = []
        time_all = []
        
        for run_results in all_runs:
            ba_all.append(run_results[model_name]['ba_scores'])
            f1_all.append(run_results[model_name]['f1_scores'])
            minf1_all.append(run_results[model_name]['minf1_scores'])
            time_all.append(run_results[model_name]['time_per_batch'])
        
        # Convert to numpy arrays
        ba_all = np.array(ba_all)  # shape: (n_runs, n_batches)
        f1_all = np.array(f1_all)
        minf1_all = np.array(minf1_all)
        time_all = np.array(time_all)
        
        # Calculate statistics
        # Average BA across all batches for each run
        avg_ba_per_run = np.mean(ba_all, axis=1)
        
        # Final BA: average over last 3 batches
        final_ba_values = np.mean(ba_all[:, -3:], axis=1)
        final_f1_values = np.mean(f1_all[:, -3:], axis=1)
        final_minf1_values = np.mean(minf1_all[:, -3:], axis=1)
        
        # AUC-BA: area under BA curve
        auc_ba_per_run = np.sum(ba_all, axis=1) / ba_all.shape[1]
        auc_ba_per_run = np.sum(ba_all, axis=1)

        # Time statistics
        time_per_batch_mean = np.mean(time_all)
        #time_mean = np.mean(time_all, axis=1)
        #time_std = np.std(time_all, axis=1)
        avg_ba_values = np.mean(ba_all, axis=1)
        
        
        time_mean = np.mean(time_all)  # ← SCALAR: mean across all runs and batches
        time_std = np.std(time_all)    # ← SCALAR: std across all runs and batches
        
        
        
        
        
        
        
        
        
        
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
    
    return all_runs, aggregated


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

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("ENHANCED WAVE-GA + EC BASELINES: Multi-Run Evaluation")
    print("Real-time BA display + Statistical significance tests")
    print("Running NEW EXPERIMENTS: Airlines Dataset + Weather Prediction")
    print("█"*80)
    
    # Store all results
    all_experiment_results = {}
    
    # Define the 2 experiments
    experiments = [
        {
            'name': 'Airlines Dataset - 100K samples (Phase 1)',
            'generator': AirlinesStream,
            'args': {
                'samples_per_batch': 200,
                'data_path': 'airlines.csv',  # Update with actual path
                'max_samples': 10000
            },
            'n_batches': 50  # 100,000 samples / 500 per batch
        },
        # {
        #     'name': 'Weather Prediction - Seasonal Drift',
        #     'generator': WeatherPredictionStream,
        #     'args': {'n_features': 20, 'n_weather_types': 4, 'samples_per_batch': 250},
        #     'n_batches': 40  # 10,000 samples total
        # }
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
