


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
- Fair comparison: same population size, generations, fitness function as Wave-GA
"""
import scikit_posthocs as sp

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.linear_model import SGDClassifier
from scipy import stats as scipy_stats
import time
import warnings
from openai._module_client import models
warnings.filterwarnings('ignore')
from river import tree, ensemble, forest,metrics

import sys
sys.path.append('/mnt/user-data/uploads')

from continual_learning_baselines_edit import ExperienceReplayWrapper, AGEMWrapper




from ultra_fast_wave_ga_buffer_without_blend import UltraFastWaveGAWithBuffer as wavegaBufferNoBlend
from ultra_fast_wave_ga_nobuffer_with_blend import UltraFastWaveGANoBufferWithBlending as wavegaNoBufferBlend
from ultra_fast_wave_ga_nobuffer_without_blend import UltraFastWaveGANoBuffer as wavegaNoBufferNoBlend
from ultra_fast_wave_ga_nowaves_nobuffer_edit import UltraFastWaveGANoWavesNoBuffer
from ultra_fast_wave_ga_nowaves_edit import UltraFastWaveGANoWaves

from ultra_fast_wave_ga_org250 import UltraFastWaveGA as StreamingWaveGA
#from streaming_wave_ga_sample_paral import StandardOnlineGA

from ultra_fast_wave_ga_nowaves_nobuffer_overall import UltraFastStandardOnlineGA
from streaming_wave_ga_sample_paral import perSampleStreamingWaveGA

from streaming_wave_ga_sample_paral import newperSampleStreamingWaveGA

#from streaming_wave_ga import StreamingWaveGA
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
#from streaming_de_classifiers import StreamingDE_Waves as UltraFastStreamingDE_Waves, StandardDE as  UltraFastStandardDE
from ultra_fast_streaming_de_classifiers_edit3 import UltraFastStreamingDE_Waves, UltraFastStandardDE

def make_wave_ga(generator, exp_name):
    print("BVAS"+exp_name)
    base_kwargs = dict(
        n_features=generator.n_features,
        population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="uniform",
        buffer_size=400,
        alpha=0.7,
    )

    if "Extreme Imbalance" in exp_name:
        print("GOGO")
        base_kwargs.update(dict(
          population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",
        buffer_size=400,
        alpha=0.9,
        ))
    elif "Recurring Drift" in exp_name:
        print("RECURRING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1000,
        alpha=0.7,
        ))
    elif "Sudden Drift" in exp_name:
        print("SUDDEN")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.30,
        mutation_strength=0.15,
        crossover_type="uniform",
        buffer_size=800,
        alpha=0.75,
        ))
    elif "Short Recurring" in exp_name:
        print("SHORTPHASE")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=4,
        mutation_rate=0.18,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1200,
        alpha=0.8,
        ))
    elif "Cycling Classes" in exp_name:
        print("CYCLING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1000,
        alpha=0.85,
        ))
    # etc...

    return StreamingWaveGA(**base_kwargs)
def make_wave_ga_b_nb(generator, exp_name):
    print("BVAS"+exp_name)
    base_kwargs = dict(
        n_features=generator.n_features,
        population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="uniform",
        buffer_size=400,
        #alpha=0.7,
    )

    if "Extreme Imbalance" in exp_name:
        print("GOGO")
        base_kwargs.update(dict(
          population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",
        buffer_size=700,
        ))
    elif "Recurring Drift" in exp_name:
        print("RECURRING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1000,
        ))
    elif "Sudden Drift" in exp_name:
        print("SUDDEN")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.30,
        mutation_strength=0.15,
        crossover_type="uniform",
        buffer_size=800,
        ))
    elif "Short Recurring" in exp_name:
        print("SHORTPHASE")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=4,
        mutation_rate=0.18,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1200,
        ))
    elif "Cycling Classes" in exp_name:
        print("CYCLING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
   
 
        ))
    # etc...

    return wavegaBufferNoBlend(**base_kwargs)

def make_wave_ga_nb_b(generator, exp_name):
    print("BVAS"+exp_name)
    base_kwargs = dict(
        n_features=generator.n_features,
        population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="uniform",
        #buffer_size=400,
        alpha=0.7,
    )

    if "Extreme Imbalance" in exp_name:

        print("GOGO")
        base_kwargs.update(dict(
          population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",

        alpha=0.9,
        ))
    elif "Recurring Drift" in exp_name:
        print("RECURRING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",

        alpha=0.7,
        ))
    elif "Sudden Drift" in exp_name:
        print("SUDDEN")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.30,
        mutation_strength=0.15,
        crossover_type="uniform",

        alpha=0.75,
        ))
    elif "Short Recurring" in exp_name:
        print("SHORTPHASE")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=4,
        mutation_rate=0.18,
        mutation_strength=0.12,
        crossover_type="arithmetic",

        alpha=0.8,
        ))
    elif "Cycling Classes" in exp_name:
        print("CYCLING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
  
        alpha=0.85,
        ))
    # etc...

    return wavegaNoBufferBlend(**base_kwargs)

def make_wave_ga_nb_nb(generator, exp_name):
    print("BVAS"+exp_name)
    base_kwargs = dict(
        n_features=generator.n_features,
        population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="uniform",
        #buffer_size=400,
        #alpha=0.7,
    )

    if "Extreme Imbalance" in exp_name:
        print("GOGO")
        base_kwargs.update(dict(
        population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",
        #buffer_size=700,
        #alpha=0.9,
        ))
    elif "Severe Drift"  in exp_name:
       if "Extreme Imbalance" in exp_name:
        print("GOGO")
        base_kwargs.update(dict(
          population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",

        ))
    elif "Recurring Drift" in exp_name:
        print("RECURRING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",

        ))
    elif "Sudden Drift" in exp_name:
        print("SUDDEN")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.30,
        mutation_strength=0.15,
        crossover_type="uniform",
   
        ))
    elif "Short Recurring" in exp_name:
        print("SHORTPHASE")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=4,
        mutation_rate=0.18,
        mutation_strength=0.12,
        crossover_type="arithmetic",
       
        ))
    elif "Cycling Classes" in exp_name:
        print("CYCLING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        
        ))
    # etc...

    return wavegaNoBufferNoBlend(**base_kwargs)



def make_wave_ga_nw(generator, exp_name):
    print("BVAS"+exp_name)
    base_kwargs = dict(
        n_features=generator.n_features,
        population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="uniform",
        buffer_size=400,
        alpha=0.7,
    )

    if "Extreme Imbalance" in exp_name:
        print("GOGO")
        base_kwargs.update(dict(
          population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",
        buffer_size=700,
        alpha=0.9,
        ))
    elif "Recurring Drift" in exp_name:
        print("RECURRING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1000,
        alpha=0.7,
        ))
    elif "Sudden Drift" in exp_name:
        print("SUDDEN")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.30,
        mutation_strength=0.15,
        crossover_type="uniform",
        buffer_size=800,
        alpha=0.75,
        ))
    elif "Short Recurring" in exp_name:
        print("SHORTPHASE")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=4,
        mutation_rate=0.18,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1200,
        alpha=0.8,
        ))
    elif "Cycling Classes" in exp_name:
        print("CYCLING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        buffer_size=1000,
        alpha=0.85,
        ))
    # etc...

    return UltraFastWaveGANoWaves(**base_kwargs)
def make_wave_ga_nw_nb(generator, exp_name):

    print("BVAS"+exp_name)
    base_kwargs = dict(
        n_features=generator.n_features,
        population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.15,
        mutation_strength=0.12,
        crossover_type="uniform",

        alpha=0.7,
    )

    if "Extreme Imbalance" in exp_name:
        print("GOGO")
        base_kwargs.update(dict(
          population_size=70,
        generations_per_wave=4,
        n_cycles=3,
        mutation_rate=0.25,
        mutation_strength=0.12,
        crossover_type="uniform",
        alpha=0.9,
        ))
    elif "Recurring Drift" in exp_name:
        print("RECURRING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        alpha=0.7,
        ))
    elif "Sudden Drift" in exp_name:
        print("SUDDEN")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=3,
        mutation_rate=0.30,
        mutation_strength=0.15,
        crossover_type="uniform",
        alpha=0.75,
        ))
    elif "Short Recurring" in exp_name:
        print("SHORTPHASE")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=8,
        n_cycles=4,
        mutation_rate=0.18,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        alpha=0.8,
        ))
    elif "Cycling Classes" in exp_name:
        print("CYCLING")
        base_kwargs.update(dict(
            population_size=70,
        generations_per_wave=10,
        n_cycles=4,
        mutation_rate=0.20,
        mutation_strength=0.12,
        crossover_type="arithmetic",
        alpha=0.85,
        ))
    # etc...

    return UltraFastWaveGANoWavesNoBuffer(**base_kwargs)


# def make_wave_ga(generator, exp_name):
#     """
#     Wave-GA (buffer + blend)  -> UltraFastWaveGA / StreamingWaveGA
#     Best overall for recurring/short recurring/cycling.
#     """
#     base_kwargs = dict(
#         n_features=generator.n_features,
#         population_size=70,
#         generations_per_wave=7,
#         n_cycles=3,
#         mutation_rate=0.14,
#         mutation_strength=0.10,
#         crossover_type="uniform",
#         buffer_size=800,
#         alpha=0.75,
#     )
#
#     if "Extreme Imbalance" in exp_name:
#         # Protect minorities strongly; moderate buffer to retain rare samples.
#         base_kwargs.update(dict(
#             generations_per_wave=5,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             buffer_size=1200,
#             alpha=0.90,
#         ))
#
#     elif "Sudden Drift" in exp_name:
#         # Forget fast: small buffer, fewer cycles, higher mutation, uniform crossover.
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=2,
#             mutation_rate=0.22,
#             mutation_strength=0.14,
#             crossover_type="uniform",
#             buffer_size=300,
#             alpha=0.60,  # lean more toward overall BA than class-focus
#         ))
#
#     elif "Recurring Drift" in exp_name:
#         # Remember both modes; reduce thrashing; keep mutation moderate; big buffer.
#         base_kwargs.update(dict(
#             population_size=40,
#
#             generations_per_wave=9,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.10,
#             crossover_type="uniform",  # smoother than arithmetic for mode switching
#             buffer_size=1800,
#             alpha=0.70,
#         ))
#
#     elif "Short Recurring" in exp_name:
#         # Very short phases + imbalance: strong memory + minority protection.
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.14,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=2200,
#             alpha=0.86,
#         ))
#
#     elif "Cycling Classes" in exp_name:
#         # Classes disappear/return: memory is crucial; keep alpha high.
#         base_kwargs.update(dict(
#             generations_per_wave=8,
#             n_cycles=3,
#             mutation_rate=0.15,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=2200,
#             alpha=0.88,
#         ))
#
#     return StreamingWaveGA(**base_kwargs)
#
#
# def make_wave_ga_b_nb(generator, exp_name):
#     """
#     Wave-GA buffer, NO blend.
#     Same as Wave-GA but without alpha (depends on your class implementation).
#     """
#     base_kwargs = dict(
#         n_features=generator.n_features,
#         population_size=70,
#         generations_per_wave=7,
#         n_cycles=3,
#         mutation_rate=0.14,
#         mutation_strength=0.10,
#         crossover_type="uniform",
#         buffer_size=800,
#     )
#
#     if "Extreme Imbalance" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=5,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             buffer_size=1200,
#         ))
#
#     elif "Sudden Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=2,
#             mutation_rate=0.22,
#             mutation_strength=0.14,
#             crossover_type="uniform",
#             buffer_size=300,
#         ))
#
#     elif "Recurring Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=8,
#             n_cycles=3,
#             mutation_rate=0.14,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=1800,
#         ))
#
#     elif "Short Recurring" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.14,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=2200,
#         ))
#
#     elif "Cycling Classes" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=8,
#             n_cycles=3,
#             mutation_rate=0.15,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=2200,
#         ))
#
#     return wavegaBufferNoBlend(**base_kwargs)
#
#
# def make_wave_ga_nb_b(generator, exp_name):
#     """
#     Wave-GA NO buffer, blend.
#     Useful when you want smoothing/class-focus without memory.
#     """
#     base_kwargs = dict(
#         n_features=generator.n_features,
#         population_size=70,
#         generations_per_wave=7,
#         n_cycles=3,
#         mutation_rate=0.16,
#         mutation_strength=0.11,
#         crossover_type="uniform",
#         alpha=0.75,
#     )
#
#     if "Extreme Imbalance" in exp_name:
#         # No buffer means minority samples are fleeting -> push alpha up.
#         base_kwargs.update(dict(
#             generations_per_wave=5,
#             n_cycles=3,
#             mutation_rate=0.20,
#             mutation_strength=0.12,
#             crossover_type="uniform",
#             alpha=0.92,
#         ))
#
#     elif "Sudden Drift" in exp_name:
#         # No buffer is actually good here: fast forgetting.
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=2,
#             mutation_rate=0.24,
#             mutation_strength=0.15,
#             crossover_type="uniform",
#             alpha=0.60,
#         ))
#
#     elif "Recurring Drift" in exp_name:
#         # Without memory, don’t over-evolve: keep moderate to avoid chasing.
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.16,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             alpha=0.72,
#         ))
#
#     elif "Short Recurring" in exp_name:
#         # Short phases + no buffer is hard: increase alpha; keep evolution light.
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=3,
#             mutation_rate=0.16,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             alpha=0.88,
#         ))
#
#     elif "Cycling Classes" in exp_name:
#         # Cycling without buffer is also hard: emphasize class-focus.
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.16,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             alpha=0.90,
#         ))
#
#     return wavegaNoBufferBlend(**base_kwargs)
#
#
# def make_wave_ga_nb_nb(generator, exp_name):
#     """
#     Wave-GA NO buffer, NO blend (pure evolutionary baseline).
#     Best suited to sudden drift among the GA variants (fast forgetting).
#     """
#     base_kwargs = dict(
#         n_features=generator.n_features,
#         population_size=70,
#         generations_per_wave=7,
#         n_cycles=3,
#         mutation_rate=0.18,
#         mutation_strength=0.12,
#         crossover_type="uniform",
#     )
#
#     if "Extreme Imbalance" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=5,
#             n_cycles=3,
#             mutation_rate=0.22,
#             mutation_strength=0.13,
#             crossover_type="uniform",
#         ))
#
#     elif "Sudden Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=2,
#             mutation_rate=0.25,
#             mutation_strength=0.15,
#             crossover_type="uniform",
#         ))
#
#     elif "Recurring Drift" in exp_name:
#         # No buffer -> avoid too much compute; won't “remember” anyway.
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.12,
#             crossover_type="uniform",
#         ))
#
#     elif "Short Recurring" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.12,
#             crossover_type="uniform",
#         ))
#
#     elif "Cycling Classes" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.12,
#             crossover_type="uniform",
#         ))
#
#     return wavegaNoBufferNoBlend(**base_kwargs)
#
#
# def make_wave_ga_nw(generator, exp_name):
#     """
#     Wave-GA NOWAVES (but still buffer+blend).
#     Since waves are removed, keep training slightly lighter to avoid thrashing.
#     """
#     base_kwargs = dict(
#         n_features=generator.n_features,
#         population_size=70,
#         generations_per_wave=6,
#         n_cycles=3,
#         mutation_rate=0.14,
#         mutation_strength=0.10,
#         crossover_type="uniform",
#         buffer_size=800,
#         alpha=0.75,
#     )
#
#     if "Extreme Imbalance" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=5,
#             n_cycles=3,
#             mutation_rate=0.18,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             buffer_size=1200,
#             alpha=0.90,
#         ))
#
#     elif "Sudden Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=2,
#             mutation_rate=0.22,
#             mutation_strength=0.14,
#             crossover_type="uniform",
#             buffer_size=300,
#             alpha=0.60,
#         ))
#
#     elif "Recurring Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.14,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=1800,
#             alpha=0.70,
#         ))
#
#     elif "Short Recurring" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=3,
#             mutation_rate=0.14,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=2200,
#             alpha=0.86,
#         ))
#
#     elif "Cycling Classes" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=7,
#             n_cycles=3,
#             mutation_rate=0.15,
#             mutation_strength=0.10,
#             crossover_type="uniform",
#             buffer_size=2200,
#             alpha=0.88,
#         ))
#
#     return UltraFastWaveGANoWaves(**base_kwargs)
#
#
#
# def make_wave_ga_nw_nb(generator, exp_name):
#     """
#     NOWAVES + NO BUFFER (blend only).
#     Weakest variant; keep compute modest and rely on alpha for stability.
#     """
#     base_kwargs = dict(
#         n_features=generator.n_features,
#         population_size=70,
#         generations_per_wave=6,
#         n_cycles=3,
#         mutation_rate=0.16,
#         mutation_strength=0.11,
#         crossover_type="uniform",
#         alpha=0.75,
#     )
#
#     if "Extreme Imbalance" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=5,
#             n_cycles=3,
#             mutation_rate=0.20,
#             mutation_strength=0.12,
#             crossover_type="uniform",
#             alpha=0.92,
#         ))
#
#     elif "Sudden Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=2,
#             mutation_rate=0.24,
#             mutation_strength=0.15,
#             crossover_type="uniform",
#             alpha=0.60,
#         ))
#
#     elif "Recurring Drift" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=3,
#             mutation_rate=0.16,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             alpha=0.72,
#         ))
#
#     elif "Short Recurring" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=3,
#             mutation_rate=0.16,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             alpha=0.88,
#         ))
#
#     elif "Cycling Classes" in exp_name:
#         base_kwargs.update(dict(
#             generations_per_wave=6,
#             n_cycles=3,
#             mutation_rate=0.16,
#             mutation_strength=0.11,
#             crossover_type="uniform",
#             alpha=0.90,
#         ))
#
#     return UltraFastWaveGANoWavesNoBuffer(**base_kwargs)
#
# # ==============================================================================
# # DATA GENERATORS
# # ==============================================================================
#
# # ==============================================================================
# # IMPROVED DATA GENERATORS - DESIGNED TO TEST MEMORY AND EVOLUTION
# # ==============================================================================
class SevereDriftStream:
    """
    Strong concept drift with sequential class arrival
    FIXED: Controlled centroid placement + high noise
    Target SNR: ~4.5 (Moderate difficulty)
    Expected BA: 0.70-0.85
    """
    
    def __init__(self, n_features=30, n_types=5, samples_per_batch=250, seed=43):
        self.n_features = n_features
        self.n_types = n_types
        self.samples_per_batch = samples_per_batch
        
        np.random.seed(seed)
        
        # FIXED: Controlled circular placement for consistent separation
        self.centroids = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types
            c = np.zeros(n_features)
            # Separation = 10 units in 2D circle
            c[0] = 10 * np.cos(angle)
            c[1] = 10 * np.sin(angle)
            # Small random variation in other dimensions
            c[2:] = np.random.randn(n_features - 2) * 0.5
            self.centroids[i] = c
        
        # Drift directions
        self.drift_dirs = {}
        for i in range(n_types):
            d = np.zeros(n_features)
            d[0] = np.random.randn() * 0.15
            d[1] = np.random.randn() * 0.15
            d[2:] = np.random.randn(n_features - 2) * 0.05
            self.drift_dirs[i] = d
        
        self.current_types = list(range(3))
        self.batch_count = 0
        
    def get_batch(self):
        self.batch_count += 1
        
        if self.batch_count == 2:
            self.current_types.append(3)
        elif self.batch_count == 4:
            self.current_types.append(4)
        
        # Apply drift
        for t in self.current_types:
            self.centroids[t] += self.drift_dirs[t]
        
        X, y = [], []
        n_per = self.samples_per_batch // len(self.current_types)
        
        for t in self.current_types:
            for _ in range(n_per):
                # High noise for difficulty
                X.append(self.centroids[t] + np.random.randn(self.n_features) * 2.2)
                y.append(t)
        
        return np.array(X), np.array(y)

class ExtremeImbalanceStream:
    """
    Extreme 92:8 imbalance with sequential rare class discovery
    STATUS: ✅ KEPT AS IS - This one is fine (stationary, PA-II should win)
    Expected BA: PA-II ~0.95-1.00, Wave-GA ~0.85-0.90
    """
    
    def __init__(self, n_features=30, n_types=6, samples_per_batch=100, seed=42):
        self.n_features = n_features
        self.n_types = n_types
        self.samples_per_batch = samples_per_batch
        
        np.random.seed(seed)
        self.centroids = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types
            c = np.zeros(n_features)
            c[0] = 12 * np.cos(angle)
            c[1] = 12 * np.sin(angle)
            c[2:] = np.random.randn(n_features - 2) * 0.4
            self.centroids[i] = c
        
        self.current_types = [0, 1]
        self.batch_count = 0
        
    def get_batch(self):
        self.batch_count += 1
        
        # Progressive class discovery
        if self.batch_count == 2:
            self.current_types.extend([2, 3])
        elif self.batch_count == 4:
            self.current_types.extend([4, 5])
        
        X, y = [], []
        
        # 92% majority class
        n_major = int(self.samples_per_batch * 0.92)
        for _ in range(n_major):
            X.append(self.centroids[0] + np.random.randn(self.n_features) * 1.3)
            y.append(0)
        
        # 8% split among rare classes
        n_minor = self.samples_per_batch - n_major
        minor_types = self.current_types[1:]
        for _ in range(n_minor):
            t = np.random.choice(minor_types)
            X.append(self.centroids[t] + np.random.randn(self.n_features) * 1.3)
            y.append(t)
        
        return np.array(X), np.array(y)


class RecurringDriftStream:
    """
    🆕 TRUE RECURRING DRIFT - This tests memory explicitly!
    
    Alternates between two distinct distributions (A ↔ B) every 5 batches.
    FIXED: Distribution B is now only SHIFTED (not rotated) for true recurring patterns.
    
    Key changes:
    - 5-batch phases (optimal for evolution convergence)
    - B is same structure as A, just shifted by 20 units (distance ~28 units)
    - Reduced noise (σ=1.5) for clearer patterns
    - 85:15 imbalance
    
    Expected: Wave-GA >> PA-II (replay buffers remember both distributions)
    Target BA: Wave-GA ~0.80-0.85, PA-II ~0.72-0.76, Gap ~+10-15%
    """
    
    def __init__(self, n_features=30, n_types=5, samples_per_batch=100, seed=43):
        self.n_features = n_features
        self.n_types = n_types
        self.samples_per_batch = samples_per_batch
        
        np.random.seed(seed)
        
        # Distribution A: Centroids at origin area
        self.centroids_A = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types
            c = np.zeros(n_features)
            c[0] = 10 * np.cos(angle)
            c[1] = 10 * np.sin(angle)
            c[2:] = np.random.randn(n_features - 2) * 0.3
            self.centroids_A[i] = c
        
        # Distribution B: Centroids SHIFTED (far from A!)
        # FIXED: Removed rotation - same structure as A, just shifted!
        # This allows memory to actually work (recurring patterns)
        self.centroids_B = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types  # SAME angle as A (no rotation!)
            c = np.zeros(n_features)
            c[0] = 10 * np.cos(angle) + 20  # Shifted +20 in x (increased from 15)
            c[1] = 10 * np.sin(angle) + 20  # Shifted +20 in y (increased from 15)
            c[2:] = np.random.randn(n_features - 2) * 0.3
            self.centroids_B[i] = c
        
        # Alternating schedule: A(5) → B(5) → A(5) → B(5)
        # INCREASED from 4 to 5 batches per phase for optimal evolution convergence
        # Total: 20 batches, 4 switches
        self.schedule = [
            'A', 'A', 'A', 'A', 'A',  # Batches 1-5 (INCREASED from 4)
            'B', 'B', 'B', 'B', 'B',  # Batches 6-10 (switch!)
            'A', 'A', 'A', 'A', 'A',  # Batches 11-15 (A returns!)
            'B', 'B', 'B', 'B', 'B',  # Batches 16-20 (B returns!)
        ]
        
        self.batch_count = 0
        
    def get_batch(self):
        # Get current distribution (A or B)
        current_dist_name = self.schedule[self.batch_count]
        centroids = self.centroids_A if current_dist_name == 'A' else self.centroids_B
        
        self.batch_count += 1
        
        X, y = [], []
        
        # IMBALANCED sampling: 85:15 ratio
        # Majority class (class 0): 85%
        n_major = int(self.samples_per_batch * 0.85)
        for _ in range(n_major):
            X.append(centroids[0] + np.random.randn(self.n_features) * 1.5)  # Reduced noise from 2.0
            y.append(0)
        
        # Minority classes (classes 1-4): 15% total
        n_minor = self.samples_per_batch - n_major
        minor_per = n_minor // (self.n_types - 1)
        for t in range(1, self.n_types):
            for _ in range(minor_per):
                X.append(centroids[t] + np.random.randn(self.n_features) * 1.5)  # Reduced noise from 2.0
                y.append(t)
        
        # Shuffle
        X, y = np.array(X), np.array(y)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


class SuddenRecurringDriftStream:
    """
    🆕 SUDDEN RECURRING DRIFT - Abrupt switches between TWO positions
    
    Alternates between positions A and B every 4 batches.
    Positions are FAR apart (28 units diagonal), requiring complete re-learning.
    
    FIXED: Reduced noise (1.5), increased to 4-batch phases for better convergence
    
    Key difference from old version:
    - OLD: D₁ → D₂ → D₃ → D₄ (never returns, PA-II adapts)
    - NEW: A ↔ B ↔ A ↔ B (recurring, PA-II forgets!)
    
    Expected: Wave-GA >> PA-II (replay remembers both positions)
    Target BA: Wave-GA ~0.78-0.85, PA-II ~0.68-0.75
    """
    
    def __init__(self, n_features=30, n_types=6, samples_per_batch=100, seed=44):
        self.n_features = n_features
        self.n_types = n_types
        self.samples_per_batch = samples_per_batch
        
        np.random.seed(seed)
        
        # Position A: Centroids around [10, 10]
        self.centroids_A = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types
            c = np.zeros(n_features)
            c[0] = 10 * np.cos(angle) + 10
            c[1] = 10 * np.sin(angle) + 10
            c[2:] = np.random.randn(n_features - 2) * 0.3
            self.centroids_A[i] = c
        
        # Position B: Centroids around [30, 30] (FAR from A!)
        self.centroids_B = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types
            c = np.zeros(n_features)
            c[0] = 10 * np.cos(angle) + 30  # +20 shift in x
            c[1] = 10 * np.sin(angle) + 30  # +20 shift in y
            c[2:] = np.random.randn(n_features - 2) * 0.3
            self.centroids_B[i] = c
        
        # Alternating schedule: A(4) ↔ B(4) ↔ A(4) ↔ B(4) ↔ A(4)
        # INCREASED from 3 to 4 batches for better convergence
        # 4 SUDDEN SWITCHES in 20 batches
        self.schedule = [
            'A', 'A', 'A', 'A',  # Batches 1-4 (INCREASED from 3)
            'B', 'B', 'B', 'B',  # Batches 5-8 (JUMP 28 units to B!)
            'A', 'A', 'A', 'A',  # Batches 9-12 (JUMP back to A!)
            'B', 'B', 'B', 'B',  # Batches 13-16 (B returns!)
            'A', 'A', 'A', 'A',  # Batches 17-20 (A returns!)
        ]
        
        self.batch_count = 0
        
    def get_batch(self):
        # Get current position (A or B)
        current_pos = self.schedule[self.batch_count]
        centroids = self.centroids_A if current_pos == 'A' else self.centroids_B
        
        self.batch_count += 1
        
        X, y = [], []
        
        # 85% majority
        n_major = int(self.samples_per_batch * 0.85)
        for _ in range(n_major):
            X.append(centroids[0] + np.random.randn(self.n_features) * 1.5)  # Reduced from 2.2
            y.append(0)
        
        # 15% minorities
        n_minor = self.samples_per_batch - n_major
        minor_per = n_minor // (self.n_types - 1)
        for t in range(1, self.n_types):
            for _ in range(minor_per):
                X.append(centroids[t] + np.random.randn(self.n_features) * 1.5)  # Reduced from 2.2
                y.append(t)
        
        X, y = np.array(X), np.array(y)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


class ShortRecurringPhasesStream:
    """
    🆕 RAPID RECURRING DRIFT - Very short phases (2 batches)
    
    Alternates every 2 batches between distributions.
    PA-II can't fully adapt in 2 batches before distribution changes again!
    
    Expected: Wave-GA >> PA-II (memory crucial for short phases)
    Target BA: Wave-GA ~0.84-0.88, PA-II ~0.68-0.75
    """
    
    def __init__(self, n_features=30, n_types=6, samples_per_batch=100, seed=45):
        self.n_features = n_features
        self.n_types = n_types
        self.samples_per_batch = samples_per_batch
        
        np.random.seed(seed)
        
        # Three distinct distributions (A, B, C)
        self.distributions = {}
        
        for dist_name, offset in [('A', [0, 0]), ('B', [12, 12]), ('C', [-10, 10])]:
            centroids = {}
            for i in range(n_types):
                angle = (2 * np.pi * i) / n_types
                c = np.zeros(n_features)
                c[0] = 9 * np.cos(angle) + offset[0]
                c[1] = 9 * np.sin(angle) + offset[1]
                c[2:] = np.random.randn(n_features - 2) * 0.3
                centroids[i] = c
            self.distributions[dist_name] = centroids
        
        # Rapid alternating schedule (2 batches per phase)
        # A(2) → B(2) → C(2) → A(2) → B(2) → C(2) → A(2) → B(2) → C(2) → A(2)
        self.schedule = [
            'A', 'A',  # 1-2
            'B', 'B',  # 3-4
            'C', 'C',  # 5-6
            'A', 'A',  # 7-8 (A returns!)
            'B', 'B',  # 9-10 (B returns!)
            'C', 'C',  # 11-12 (C returns!)
            'A', 'A',  # 13-14
            'B', 'B',  # 15-16
            'C', 'C',  # 17-18
            'A', 'A'   # 19-20
        ]
        
        self.batch_count = 0
        
    def get_batch(self):
        # Get current distribution
        current_dist_name = self.schedule[self.batch_count]
        centroids = self.distributions[current_dist_name]
        
        self.batch_count += 1
        
        X, y = [], []
        
        # Imbalanced: 88% class 0, 12% others
        n_major = int(self.samples_per_batch * 0.88)
        for _ in range(n_major):
            X.append(centroids[0] + np.random.randn(self.n_features) * 1.8)
            y.append(0)
        
        n_minor = self.samples_per_batch - n_major
        minor_per = n_minor // (self.n_types - 1)
        for t in range(1, self.n_types):
            for _ in range(minor_per):
                X.append(centroids[t] + np.random.randn(self.n_features) * 1.8)
                y.append(t)
        
        X, y = np.array(X), np.array(y)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]


class CyclingClassesStream:
    """
    🆕 CLASS CYCLING - Classes appear and disappear!
    
    UPDATED: Shorter phases (2 batches), higher imbalance (88:12), 3 groups
    This creates much stronger memory pressure on PA-II.
    
    Expected: Wave-GA >> PA-II (replay remembers all classes)
    Target BA: Wave-GA ~0.84-0.86, PA-II ~0.65-0.70
    """
    
    def __init__(self, n_features=30, n_types=6, samples_per_batch=100, seed=46):
        self.n_features = n_features
        self.n_types = n_types
        self.samples_per_batch = samples_per_batch
        
        np.random.seed(seed)
        
        # Create centroids for ALL 6 classes
        self.centroids = {}
        for i in range(n_types):
            angle = (2 * np.pi * i) / n_types
            c = np.zeros(n_features)
            c[0] = 11 * np.cos(angle)
            c[1] = 11 * np.sin(angle)
            c[2:] = np.random.randn(n_features - 2) * 0.3
            self.centroids[i] = c
        
        # Class cycling schedule - 3 GROUPS, 2-BATCH PHASES
        # Group A: Classes [0, 1, 2]
        # Group B: Classes [0, 3, 4]
        # Group C: Classes [0, 5]
        
        self.phase_schedule = [
            [0, 1, 2],     # Batches 1-2 (Group A)
            [0, 1, 2],
            
            [0, 3, 4],     # Batches 3-4 (Group B, A disappears!)
            [0, 3, 4],
            
            [0, 5],        # Batches 5-6 (Group C, B disappears!)
            [0, 5],
            
            [0, 1, 2],     # Batches 7-8 (A returns!)
            [0, 1, 2],
            
            [0, 3, 4],     # Batches 9-10 (B returns!)
            [0, 3, 4],
            
            [0, 5],        # Batches 11-12 (C returns!)
            [0, 5],
            
            [0, 1, 2],     # Batches 13-14
            [0, 1, 2],
            
            [0, 3, 4],     # Batches 15-16
            [0, 3, 4],
            
            [0, 5],        # Batches 17-18
            [0, 5],
            
            [0, 1, 2],     # Batches 19-20
            [0, 1, 2],
        ]
        
        self.batch_count = 0
        
    def get_batch(self):
        # Get active classes for this batch
        active_classes = self.phase_schedule[self.batch_count]
        
        self.batch_count += 1
        
        X, y = [], []
        
        # INCREASED IMBALANCE: 88:12 (was 85:15)
        n_major = int(self.samples_per_batch * 0.88)
        for _ in range(n_major):
            X.append(self.centroids[0] + np.random.randn(self.n_features) * 1.5)
            y.append(0)
        
        # Minorities (12% total)
        n_minor = self.samples_per_batch - n_major
        minor_classes = [c for c in active_classes if c != 0]
        minor_per = n_minor // len(minor_classes)
        
        for t in minor_classes:
            for _ in range(minor_per):
                X.append(self.centroids[t] + np.random.randn(self.n_features) * 1.5)
                y.append(t)
        
        X, y = np.array(X), np.array(y)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

#     """Experiment 2: Strong drift"""
#     def __init__(self, n_features=30, n_types=5, samples_per_batch=250, seed=43):
#         self.n_features = n_features
#         self.n_types = n_types
#         self.samples_per_batch = samples_per_batch
#
#         np.random.seed(seed)
#         self.centroids = {}
#         for i in range(n_types):
#             angle = (2 * np.pi * i) / n_types
#             c = np.zeros(n_features)
#             c[0] = 9 * np.cos(angle)
#             c[1] = 9 * np.sin(angle)
#             c[2:] = np.random.randn(n_features - 2) * 0.5
#             self.centroids[i] = c
#
#         self.drift_dirs = {}
#         for i in range(n_types):
#             d = np.zeros(n_features)
#             d[0] = np.random.randn() * 0.15
#             d[1] = np.random.randn() * 0.15
#             d[2:] = np.random.randn(n_features - 2) * 0.05
#             self.drift_dirs[i] = d
#
#         self.current_types = [0, 1]
#         self.batch_count = 0
#
#     def get_batch(self):
#         self.batch_count += 1
#
#         if self.batch_count == 2:
#             self.current_types.append(2)
#         elif self.batch_count == 4:
#             self.current_types.extend([3, 4])
#
#         # Apply drift
#         for i in range(self.n_types):
#             self.centroids[i] += self.drift_dirs[i]
#
#         X, y = [], []
#
#         n_per_class = self.samples_per_batch // len(self.current_types)
#         for t in self.current_types:
#             for _ in range(n_per_class):
#                 X.append(self.centroids[t] + np.random.randn(self.n_features) * 2.5)
#                 y.append(t)
#
#         return np.array(X), np.array(y)
#
#
# class CombinedStream:
#     """Experiment 3: Imbalance + drift"""
#     def __init__(self, n_features=25, n_types=6, samples_per_batch=250, seed=44):
#         self.n_features = n_features
#         self.n_types = n_types
#         self.samples_per_batch = samples_per_batch
#
#         np.random.seed(seed)
#         self.centroids = {}
#         for i in range(n_types):
#             angle = (2 * np.pi * i) / n_types
#             c = np.zeros(n_features)
#             c[0] = 9 * np.cos(angle)
#             c[1] = 9 * np.sin(angle)
#             c[2:] = np.random.randn(n_features - 2) * 0.5
#             self.centroids[i] = c
#
#         self.drift_dirs = {}
#         for i in range(n_types):
#             d = np.zeros(n_features)
#             d[0] = np.random.randn() * 0.12
#             d[1] = np.random.randn() * 0.12
#             d[2:] = np.random.randn(n_features - 2) * 0.04
#             self.drift_dirs[i] = d
#
#         self.current_types = [0, 1]
#         self.batch_count = 0
#
#     def get_batch(self):
#         self.batch_count += 1
#
#         if self.batch_count == 2:
#             self.current_types.extend([2, 3])
#         elif self.batch_count == 4:
#             self.current_types.extend([4, 5])
#
#         # Apply drift
#         for i in range(self.n_types):
#             self.centroids[i] += self.drift_dirs[i]
#
#         X, y = [], []
#
#         # 75% majority
#         n_major = int(self.samples_per_batch * 0.75)
#         for _ in range(n_major):
#             X.append(self.centroids[0] + np.random.randn(self.n_features) * 2.2)
#             y.append(0)
#
#         # 25% minority
#         n_minor = self.samples_per_batch - n_major
#         minor_types = self.current_types[1:]
#         if len(minor_types) > 0:
#             n_per_minor = n_minor // len(minor_types)
#             for t in minor_types:
#                 for _ in range(n_per_minor):
#                     X.append(self.centroids[t] + np.random.randn(self.n_features) * 2.2)
#                     y.append(t)
#
#         return np.array(X), np.array(y)
#
#
# class VeryHardOverlappingStream:
#     """Experiment 4: Severe overlap"""
#     def __init__(self, n_features=30, n_types=6, samples_per_batch=250, seed=45):
#         self.n_features = n_features
#         self.n_types = n_types
#         self.samples_per_batch = samples_per_batch
#
#         np.random.seed(seed)
#         self.centroids = {}
#         for i in range(n_types):
#             angle = (2 * np.pi * i) / n_types
#             c = np.zeros(n_features)
#             c[0] = 5.5 * np.cos(angle)  # Tighter spacing
#             c[1] = 5.5 * np.sin(angle)
#             c[2:] = np.random.randn(n_features - 2) * 0.3
#             self.centroids[i] = c
#
#         self.current_types = [0, 1, 2]
#         self.batch_count = 0
#
#     def get_batch(self):
#         self.batch_count += 1
#
#         if self.batch_count == 3:
#             self.current_types.extend([3, 4, 5])
#
#         X, y = [], []
#         n_per_class = self.samples_per_batch // len(self.current_types)
#
#         for t in self.current_types:
#             for _ in range(n_per_class):
#                 X.append(self.centroids[t] + np.random.randn(self.n_features) * 2.8)
#                 y.append(t)
#
#         return np.array(X), np.array(y)
#
#
# class ExtremeNonStationaryStream:
#     """Experiment 5: Drift reversals"""
#     def __init__(self, n_features=30, n_types=5, samples_per_batch=250, seed=46):
#         self.n_features = n_features
#         self.n_types = n_types
#         self.samples_per_batch = samples_per_batch
#
#         np.random.seed(seed)
#         self.initial_centroids = {}
#         for i in range(n_types):
#             angle = (2 * np.pi * i) / n_types
#             c = np.zeros(n_features)
#             c[0] = 8 * np.cos(angle)
#             c[1] = 8 * np.sin(angle)
#             c[2:] = np.random.randn(n_features - 2) * 0.4
#             self.initial_centroids[i] = c.copy()
#
#         self.centroids = {i: c.copy() for i, c in self.initial_centroids.items()}
#
#         self.drift_dirs = {}
#         for i in range(n_types):
#             d = np.zeros(n_features)
#             d[0] = np.random.randn() * 0.18
#             d[1] = np.random.randn() * 0.18
#             d[2:] = np.random.randn(n_features - 2) * 0.06
#             self.drift_dirs[i] = d
#
#         self.batch_count = 0
#
#     def get_batch(self):
#         self.batch_count += 1
#
#         # Drift reversals
#         if self.batch_count in [3, 6]:
#             for i in range(self.n_types):
#                 self.drift_dirs[i] *= -1
#
#         # Apply drift
#         for i in range(self.n_types):
#             self.centroids[i] += self.drift_dirs[i]
#
#         X, y = [], []
#
#         # Dynamic imbalance
#         if self.batch_count <= 2:
#             ratios = [0.93, 0.02, 0.02, 0.02, 0.01]
#         else:
#             ratios = [0.78, 0.06, 0.06, 0.06, 0.04]
#
#         for i, ratio in enumerate(ratios):
#             n_samples = int(self.samples_per_batch * ratio)
#             for _ in range(n_samples):
#                 X.append(self.centroids[i] + np.random.randn(self.n_features) * 2.2)
#                 y.append(i)
#
#         return np.array(X), np.array(y)


# ==============================================================================
# MODEL WRAPPERS
# ==============================================================================

class RiverHoeffdingTreeWrapper:
    def __init__(self, n_features, seed=42):
        self.model = tree.HoeffdingTreeClassifier(grace_period= 200,leaf_prediction='mc', delta=0.01, tau=0.05)
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
        # self.model = forest.ARFClassifier(n_models=30, seed=seed, grace_period=150, delta=0.00001)
        #self.model = forest.ARFClassifier(n_models=30, seed=seed, grace_period= 200,leaf_prediction='mc',split_confidence= 0.01,delta=0.01)
        self.model = forest.ARFClassifier(
        n_models=30,
        seed=seed,
        grace_period=200,  # Increased from 150  
        delta=0.01,  # Changed from 0.00001 - CRITICAL FIX
        metric=metrics.BalancedAccuracy()  # CRITICAL for imbalance
    )

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

    # 'grace_period': 200,
    # 'leaf_prediction': 'mc',
    # 'split_confidence': 0.01
class RiverLeveragingBaggingWrapper:
    def __init__(self, n_features, seed=42):
        base_model = tree.HoeffdingTreeClassifier(grace_period= 200,leaf_prediction='mc',delta=0.01, tau=0.05)
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
        base_model = tree.HoeffdingTreeClassifier(grace_period= 200,leaf_prediction='mc', delta=0.01, tau=0.05)
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
        base_model = tree.HoeffdingTreeClassifier(grace_period= 200,leaf_prediction='mc', tau=0.05)
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
# SINGLE RUN EXPERIMENT
# ==============================================================================

def run_single_experiment(generator, name, n_batches=20, seed=42, run_number=1, total_runs=5, suppress_output=True):
    """Run one complete experiment with given seed"""
    
    import sys
    import os
    
    # Initialize models with seed
    models = {}
    print(name)
    
    models['StandardGA']=UltraFastStandardOnlineGA(n_features=generator.n_features,    population_size=70,
        generations_per_wave=7,
        n_cycles=3,

        mutation_rate=0.15,
        mutation_strength=0.12,)

 
    #


    np.random.seed(seed)
    models['Wave-GA'] = make_wave_ga(generator, name)
    
    np.random.seed(seed)
    models['Wave-GA-buffer-noblend'] = make_wave_ga_b_nb(generator, name)
    
        
    np.random.seed(seed)
    models['Wave-GA-nobuffer-noblend'] = make_wave_ga_nb_nb(generator, name)
    
           
    np.random.seed(seed)
    models['Wave-GA-nobuffer-blend'] = make_wave_ga_nb_b(generator, name)
    np.random.seed(seed)
    
    models['Wave-GA-nowaves'] = make_wave_ga_nw(generator, name)
    np.random.seed(seed)
    models['Wave-GA-nowaves-nobuffer'] = make_wave_ga_nw_nb(generator, name)

    print("HERE")
    # if suppress_output:
    #     original_stdout = sys.stdout
    #     sys.stdout = open(os.devnull, 'w')
    # np.random.seed(seed)
    # print(name)
    # ==========================================================================
    # EVOLUTIONARY COMPUTATION BASELINES (for GECCO comparison)
    # ==========================================================================
    
    # DE + Waves: Differential Evolution with same wave structure as Wave-GA
    np.random.seed(seed)
    models['DE+Waves'] = UltraFastStreamingDE_Waves(
        n_features=generator.n_features,
        population_size=70,              # Same as Wave-GA
                 # Same as Wave-GA
        F=0.8,                           # Standard DE mutation factor
        CR=0.9,                          # Standard DE crossover rate
              generations_per_wave=8,          # Same as Wave-GA
        n_cycles=3,     
        
        alpha=0.7                        # Same fitness blending as Wave-GA
    )
    
    # Standard DE (no waves): Ablation baseline to show wave structure matters
    np.random.seed(seed)
    models['StandardDE'] = UltraFastStandardDE(
        n_features=generator.n_features,
        population_size=70,              # Same as Wave-GA
        generations=24*4,                  # Total: 3 cycles × 8 gen/wave (equiv to Wave-GA)
                #generations_per_wave=8,          # Same as Wave-GA
        #n_cycles=3,     
        
        F=0.8,                           # Standard DE
        CR=0.9,                          # Standard DE
        #buffer_size_per_class=200
    )
    
    
    np.random.seed(seed)
    models['ER (Replay)'] = ExperienceReplayWrapper(
    n_features=generator.n_features,
    buffer_size=200, learning_rate=0.01,
    update_style='pa', random_state=seed
        )

    np.random.seed(seed)
    models['A-GEM'] = AGEMWrapper(
    n_features=generator.n_features,
    buffer_size=200, learning_rate=0.01,
    memory_strength=0.5, random_state=seed
    )
    # np.random.seed(seed)
    # models['WaveIm'] = ImprovedStreamingWaveGA(
    #     n_features=generator.n_features,
    #     population_size=25,
    #     generations_per_wave=8,
    #     n_cycles=5,
    #     mutation_rate=0.15,
    #     mutation_strength=0.12,
    #     buffer_size_per_class=200
    # )
     
    # np.random.seed(42)
    # models['CWI'] = CorrectedImprovedWaveGA(
    #     n_features=generator.n_features,
    #     population_size=25,
    #     generations_per_wave=8,
    #     n_cycles=5,
    #     lambda_reg=0.01,
    #     val_split=0.25,
    #     patience=3,
    #     augment_rare=True
    # )
    #
    # models['perSampleWave-GA'] = perSampleStreamingWaveGA(
    #     n_features=generator.n_features,
    #     population_size=25,
    #     generations_per_wave=8,
    #     n_cycles=3,
    #     mutation_rate=0.15,
    #     mutation_strength=0.12
    # )
    #
    # np.random.seed(42)
    # models['tunedSampleWaveGa']=newperSampleStreamingWaveGA(
    #     n_features=generator.n_features,                 # your value
    #     population_size=25,
    #     generations_per_wave=8,
    #     n_cycles=3,
    #     mutation_rate=0.1,
    #     mutation_strength=0.1,
    #
    #     # ---- imbalance-related (strong protection for minority) ----
    #     class_weight_power=1.0,        # was 0.5 → stronger upweighting of rare classes
    #     majority_freq_threshold=0.7,   # treat anything >70% as majority
    #     majority_update_prob=0.05,     # only 5% of majority samples trigger micro-evolution
    #
    #     # ---- buffer-related (minority gets more memory) ----
    #     base_buffer_size=1000,
    #     min_buffer_size=300,
    #     max_buffer_size=3000,
    #
    #     # ---- drift-related (pretty sensitive) ----
    #     drift_window_size=40,          # history length for BA
    #     drift_min_window=15,
    #     drift_drop_threshold=0.08,     # trigger drift if BA drops by > 8 points
    #     drift_mutation_factor=2.0,     # double mutation in drift mode
    #     drift_generations_factor=1.5,  # 50% more generations in drift mode
    #     drift_cooldown=60              # stay in drift mode for ~60 batch updates
    # )
    # np.random.seed(42)
    # models['StandardGA']=StandardOnlineGA(n_features=generator.n_features)
    # np.random.seed(42)
    #



    # PA-II is typically best - always include
    models['PA-II (C=1.0)'] = LinearStreamingWrapper(
        PassiveAggressiveClassifier(C=1.0, variant='PA-II', random_state=seed),
        name='PA-II (C=1.0)'
    )
    # np.random.seed(42)
    #
    # models['PA-II (C=10)'] = LinearStreamingWrapper(
    #     PassiveAggressiveClassifier(C=10.0, variant='PA-II', random_state=seed),
    #     name='PA-II (C=10)'
    # )
    # np.random.seed(42)

    # AROW - particularly good for noisy/imbalanced data
    models['AROW (r=1.0)'] = LinearStreamingWrapper(
        AROWClassifier(r=1.0, random_state=seed),
        name='AROW (r=1.0)'
    )
    
    np.random.seed(seed)
    
    models['AROW (r=0.1)'] = LinearStreamingWrapper(
        AROWClassifier(r=0.1, random_state=seed),
        name='AROW (r=0.1)'
    )
    np.random.seed(seed)
  
    # Confidence-Weighted Learning
    models['CW (h=0.9)'] = LinearStreamingWrapper(
        ConfidenceWeightedClassifier(eta=0.9, random_state=seed),
        name='CW (h=0.9)'
    )
    np.random.seed(seed)
    
    # Second-Order Perceptron
    models['SOP'] = LinearStreamingWrapper(
        SecondOrderPerceptron(a=1.0, random_state=seed),
        name='SOP'
    )

    np.random.seed(seed)

    # =========================================================================
    # SIMPLE LINEAR BASELINE (for reference)
    # =========================================================================
    # models['SGD-Online'] = SGDClassifier(
    #     loss='log_loss', max_iter=10, random_state=seed
    # )
    np.random.seed(seed)
    models['ARF (SOTA)'] = RiverAdaptiveRandomForestWrapper(n_features=generator.n_features, seed=seed)
    
    np.random.seed(seed)
    models['Leveraging Bagging'] = RiverLeveragingBaggingWrapper(n_features=generator.n_features, seed=seed)
    
    np.random.seed(seed)
    models['Hoeffding Tree'] = RiverHoeffdingTreeWrapper(n_features=generator.n_features, seed=seed)
    
    np.random.seed(seed)
    models['ADWIN Bagging'] = RiverADWINBaggingWrapper(n_features=generator.n_features, seed=seed)
    
    np.random.seed(seed)
    models['Online Bagging'] = RiverOnlineBaggingWrapper(n_features=generator.n_features, seed=seed)
    
    # np.random.seed(42)
    # models['SGD-Online'] = SimpleOnlineClassifier(n_features=generator.n_features, seed=seed)
    #

    # Restore stdout after initialization

    print("HERE")
    if suppress_output:
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    np.random.seed(seed)
    print(name)
    results = {n: {'ba': [], 'f1': [], 'min_f1': [], 'time': []} for n in models}
    
    model_names = list(models.keys())
    
    for batch in range(n_batches):
        X_batch, y_batch = generator.get_batch()
    
        # Prequential: test then train on the same batch
        X_te, y_te = X_batch, y_batch
        X_tr, y_tr = X_batch, y_batch
        
        for model_idx, (model_name, model) in enumerate(models.items()):
            
            # Suppress output during actual training/testing
            if suppress_output:
                sys.stdout = open(os.devnull, 'w')
            
            # Test
            if batch == 0:
                y_pred = np.zeros(len(y_te), dtype=int)
            else:
                y_pred = model.predict(X_te)
            
            # Metrics
            ba = balanced_accuracy_score(y_te, y_pred)
            f1_macro = f1_score(y_te, y_pred, average='macro', zero_division=0)
            
            minority_mask = y_te != 0
            if np.sum(minority_mask) > 0:
                min_f1 = f1_score(y_te[minority_mask], y_pred[minority_mask], average='macro', zero_division=0)
            else:
                min_f1 = 0.0
            
            # Train
            t0 = time.time()
            model.partial_fit(X_tr, y_tr)
            train_time = time.time() - t0
            
            # Restore stdout
            if suppress_output:
                sys.stdout.close()
                sys.stdout = original_stdout
            
            results[model_name]['ba'].append(ba)
            results[model_name]['f1'].append(f1_macro)
            results[model_name]['min_f1'].append(min_f1)
            results[model_name]['time'].append(train_time)
            
            # Show progress with BA
            progress = f"  Trial {run_number}/{total_runs} (seed={seed}): B{batch+1}/{n_batches} [{model_idx+1}/{len(models)}] {model_name:<22} BA:{ba:.3f} MinF1:{min_f1:.3f} time:{train_time:.3f} "
            print(progress)
    
    return results


# ==============================================================================
# MULTI-RUN EXPERIMENT
# ==============================================================================

def run_multirun_experiment(generator_class, generator_args, exp_name, n_runs=30, n_batches=20):
    """
    Run experiment multiple times with different seeds
    
    Returns:
        all_runs: List of results dicts (one per run)
        aggregated: Dict with mean/std for each model
    """
    
    print(f"\n{'='*80}")
    print(f"{exp_name.upper()}")
    print(f"{'='*80}")
    print(f"Running {n_runs} independent trials (7 models × 6 batches each)...")
    
    all_runs = []
    trial_times = []
    
    for run in range(n_runs):
        seed = 42 + run * 10  # Different seeds: 42, 52, 62, 72, 82
        
        print(f"\n  === Trial {run+1}/{n_runs} (seed={seed}) ===")
        
        trial_start = time.time()
        
        # Create fresh generator with new seed
        generator = generator_class(**{**generator_args, 'seed': seed})
        
        # Run experiment (with BA display)
        results = run_single_experiment(generator, exp_name, n_batches=n_batches, seed=seed, 
                                       run_number=run+1, total_runs=n_runs, suppress_output=True)
        all_runs.append(results)
        
        trial_time = time.time() - trial_start
        trial_times.append(trial_time)
        
        print(f"  ✓ Trial {run+1} completed in {trial_time:.1f}s")
    
    avg_time = np.mean(trial_times)
    print(f"\nAll trials complete! Average: {avg_time:.1f}s per trial")
    
    # Aggregate across runs
    aggregated = aggregate_runs(all_runs)
    
    return all_runs, aggregated


def aggregate_runs(all_runs):
    """Aggregate multiple runs into mean ± std"""
    
    n_runs = len(all_runs)
    model_names = list(all_runs[0].keys())
    n_batches = len(all_runs[0][model_names[0]]['ba'])
    
    aggregated = {}
    
    for model_name in model_names:
        # Collect data across runs
        ba_all = np.array([run[model_name]['ba'] for run in all_runs])  # Shape: (n_runs, n_batches)
        f1_all = np.array([run[model_name]['f1'] for run in all_runs])
        minf1_all = np.array([run[model_name]['min_f1'] for run in all_runs])
        time_all = np.array([run[model_name]['time'] for run in all_runs])
        
        # Per-batch statistics
        ba_per_batch_mean = np.mean(ba_all, axis=0)
        ba_per_batch_std = np.std(ba_all, axis=0)
        
        f1_per_batch_mean = np.mean(f1_all, axis=0)
        f1_per_batch_std = np.std(f1_all, axis=0)
        
        minf1_per_batch_mean = np.mean(minf1_all, axis=0)
        minf1_per_batch_std = np.std(minf1_all, axis=0)
        
        # Overall statistics (average over all batches and runs)
        avg_ba_per_run = np.mean(ba_all, axis=1)  # Average over batches for each run
        avg_ba = np.mean(avg_ba_per_run)          # Average over runs
        avg_ba_std = np.std(avg_ba_per_run)
        
        # AUC (area under curve) - sum of BA across batches
        auc_ba_per_run = np.sum(ba_all, axis=1)
        auc_ba = np.mean(auc_ba_per_run)
        auc_ba_std = np.std(auc_ba_per_run)
        
        # Final performance (last 3 batches)
        final_ba_per_run = np.mean(ba_all[:, -5:], axis=1)
        final_ba = np.mean(final_ba_per_run)
        final_ba_std = np.std(final_ba_per_run)
        
        final_f1_per_run = np.mean(f1_all[:, -5:], axis=1)
        final_f1 = np.mean(final_f1_per_run)
        final_f1_std = np.std(final_f1_per_run)
        
        final_minf1_per_run = np.mean(minf1_all[:, -5:], axis=1)
        final_minf1 = np.mean(final_minf1_per_run)
        final_minf1_std = np.std(final_minf1_per_run)
        
        # Runtime statistics
        time_per_batch_mean = np.mean(time_all, axis=0)
        time_mean = np.mean(time_all)
        time_std = np.std(time_all)
        
        # Store raw BA values for t-tests (last 3 batches, all runs)
        final_ba_values = np.mean(ba_all[:, -5:], axis=1)  # Shape: (n_runs,)
        final_f1_values = np.mean(f1_all[:, -5:], axis=1)
        final_minf1_values = np.mean(minf1_all[:, -5:], axis=1)
        
        avg_ba_values = np.mean(ba_all, axis=1)
        avg_f1_values = np.mean(f1_all, axis=1)
        avg_minf1_values = np.mean(minf1_all, axis=1)
        
        aggregated[model_name] = {
            # Per-batch arrays
            'ba_per_batch_mean': ba_per_batch_mean,
            'ba_per_batch_std': ba_per_batch_std,
            'f1_per_batch_mean': f1_per_batch_mean,
            'f1_per_batch_std': f1_per_batch_std,
            'minf1_per_batch_mean': minf1_per_batch_mean,
            'minf1_per_batch_std': minf1_per_batch_std,
            
            # Aggregate metrics
            'avg_ba': avg_ba,
            'avg_ba_std': avg_ba_std,
            'auc_ba': auc_ba,
            'auc_ba_std': auc_ba_std,
            'final_ba': final_ba,
            'final_ba_std': final_ba_std,
            
            # Same for F1
            'avg_f1': np.mean(f1_all),
            'avg_f1_std': np.std(np.mean(f1_all, axis=1)),
            'final_f1': np.mean(np.mean(f1_all[:, -5:], axis=1)),
            'final_f1_std': np.std(np.mean(f1_all[:, -5:], axis=1)),
            
            # Same for MinF1
            'avg_minf1': np.mean(minf1_all),
            'avg_minf1_std': np.std(np.mean(minf1_all, axis=1)),
            'final_minf1': np.mean(np.mean(minf1_all[:, -5:], axis=1)),
            'final_minf1_std': np.std(np.mean(minf1_all[:, -5:], axis=1)),
            
            # Runtime
            'time_per_batch_mean': time_per_batch_mean,
            'time_mean': time_mean,
            'time_std': time_std,
            
            # Raw values for t-tests
            'final_ba_values': final_ba_values,
            'final_f1_values': final_f1_values,
            'final_minf1_values': final_minf1_values,
            'final_avg_ba_values': avg_ba_values,
            'final_avg_f1_values': avg_f1_values,
            'final_avg_minf1_values': avg_minf1_values,
        }
    
    return aggregated


# ==============================================================================
# REPORTING FUNCTIONS
# ==============================================================================

def print_aggregate_results(aggregated, exp_name):
    """Print aggregated results with mean ± std"""
    
    print(f"\n{'='*160}")
    print(f"{exp_name} - AGGREGATED RESULTS (Mean ± Std over 30 runs)")
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
        time_str = f"{data['time_mean']:.4f}±{data['time_std']:.4f}"
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

    # --------------------------
    # (A) Your existing t-tests
    # --------------------------
    for model_name in model_names:
        if model_name == 'Wave-GA':
            continue
        
        for metric in ['ba', 'f1', 'minf1', 'avg_ba', 'avg_minf1']:
            wave_ga_values = aggregated['Wave-GA'][f'final_{metric}_values']
            other_values   = aggregated[model_name][f'final_{metric}_values']
            
            if len(wave_ga_values) > 1 and len(other_values) > 1:
                t_stat, p_value = scipy_stats.ttest_ind(wave_ga_values, other_values)
                
                if p_value < 0.001:
                    sig = "*** (p<0.001)"
                elif p_value < 0.01:
                    sig = "** (p<0.01)"
                elif p_value < 0.05:
                    sig = "* (p<0.05)"
                else:
                    sig = "n.s."
                
                wave_mean = np.mean(wave_ga_values)
                other_mean = np.mean(other_values)
                direction = ">" if wave_mean > other_mean else "<"
                
                comparison = f"Wave-GA ({wave_mean:.4f}) {direction} {model_name} ({other_mean:.4f})"
                metric_name = metric.upper() if metric != 'minf1' else 'MinF1'
                print(f"{comparison:<50} {metric_name:<12} {p_value:<15.6f} {sig:<20}")
    
    print(f"\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, n.s. = not significant")
    print(f"{'='*120}")

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
    ablation_candidates = [m for m in aggregated.keys() if m.lower().startswith("wave-ga") and m != "Wave-GA"]
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


if __name__ == "__main__":
    print("\n" + "█"*80)
    print("ENHANCED WAVE-GA + EC BASELINES: Multi-Run Evaluation")
    print("Real-time BA display + Statistical significance tests")
    print("Running ALL 5 EXPERIMENTS with GA, DE+Waves, and StandardDE")
    print("█"*80)
    
    # ========== ADD THIS WARMUP CODE HERE (at the very start) ==========
    print("\n" + "="*80)
    print("WARMING UP NUMBA (Pre-compiling WaveGA)...")
    print("="*80)
    
    import time
    import numpy as np
    
    # Import WaveGA
   # from ultra_fast_wave_ga_TRUE_NUMBA import UltraFastWaveGA_Numba as StreamingWaveGA
    
    # Create dummy model (small size for speed)
    warmup_model = StreamingWaveGA(
        n_features=30,  # Match your experiments
        population_size=10,  # Small
        generations_per_wave=2,  # Minimal
        n_cycles=1,  # Single cycle
        seed=999
    )
    
    # Generate dummy data and run warmup
    X_warmup = np.random.randn(50, 30)
    y_warmup = np.random.randint(0, 3, 50)
    
    warmup_start = time.time()
    warmup_model.partial_fit(X_warmup, y_warmup)  # First call - compiles
    warmup_model.partial_fit(X_warmup, y_warmup)  # Second call - verify
    warmup_time = time.time() - warmup_start
    
    print(f"✓ Warmup complete in {warmup_time:.2f}s")
    print(f"  All WaveGA calls will now be fast!")
    print("="*80 + "\n")
    # Store all results
    all_experiment_results = {}
    
    # Define all 5 experiments
    experiments = [
           {
            'name': 'Severe Drift',
            'generator': SevereDriftStream,
            'args': {'n_features': 30, 'n_types': 5, 'samples_per_batch': 100}
        },
        {
            'name': 'Recurring Drift (3-batch phases)',
            'generator': RecurringDriftStream,
            'args': {'n_features': 30, 'n_types': 5, 'samples_per_batch': 100}
        },
        {
            'name': 'Sudden Drift (Abrupt jumps)',
            'generator': SuddenRecurringDriftStream,
            'args': {'n_features': 30, 'n_types': 6, 'samples_per_batch': 100}
        },
   
        {
            'name': 'Short Recurring Phases (2-batch)',
            'generator': ShortRecurringPhasesStream,
            'args': {'n_features': 30, 'n_types': 6, 'samples_per_batch': 100}
        },
        {
            'name': 'Cycling Classes',
            'generator': CyclingClassesStream,
            'args': {'n_features': 30, 'n_types': 6, 'samples_per_batch': 100}
        },
             {
            'name': 'Extreme Imbalance (92:8)',
            'generator': ExtremeImbalanceStream,
            'args': {'n_features': 30, 'n_types': 6, 'samples_per_batch': 100}
        },
    
    ]
    
    # Run each experiment
    for exp_idx, exp in enumerate(experiments, 1):
        print(f"\n{'#'*80}")
        print(f"EXPERIMENT {exp_idx}/5: {exp['name']}")
        print(f"{'#'*80}")
        
        all_runs, aggregated = run_multirun_experiment(
            exp['generator'],
            exp['args'],
            exp['name'],
            n_runs=30,
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
    print("FINAL SUMMARY about FinalBa: Wave-GA Performance Across All 5 Experiments")
    print("="*100)
    print(f"{'Experiment':<35} {'Wave-GA Final BA':<20} {'Best Baseline':<20} {'Best BA':<15}")
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
        
        print(f"{exp_name:<35} {wave_str:<20} {best_name:<20} {best_str:<15}")
    
    print("="*100)
     # Final summary across all experiments
    print("\n" + "="*100)
    print("FINAL SUMMARY about avgBA: Wave-GA Performance Across All 5 Experiments")
    print("="*100)
    print(f"{'Experiment':<35} {'Wave-GA avg BA':<20} {'Best Baseline':<20} {'Best avg BA':<15}")
    print("-"*100)
    
    for exp in experiments:
        exp_name = exp['name']
        agg = all_experiment_results[exp_name]['aggregated']
        
        wave_ga_ba = agg['Wave-GA']['avg_ba']
        wave_ga_std = agg['Wave-GA']['avg_ba_std']
        
        # Find best baseline
        best_ba = 0.0
        best_name = ""
        for model_name, data in agg.items():
            if model_name != 'Wave-GA':
                if data['avg_ba'] > best_ba:
                    best_ba = data['avg_ba']
                    best_name = model_name
        
        wave_str = f"{wave_ga_ba:.4f}±{wave_ga_std:.4f}"
        best_str = f"{best_ba:.4f}"
        
        print(f"{exp_name:<35} {wave_str:<20} {best_name:<20} {best_str:<15}")
    print("="*100)
     # Final summary across all experiments
    print("\n" + "="*100)
    print("FINAL SUMMARY about minf1: Wave-GA Performance Across All 5 Experiments")
    print("="*100)
    print(f"{'Experiment':<35} {'Wave-GA avg minF1':<20} {'Best Baseline':<20} {'Best avg minf1':<15}")
    print("-"*100)
    
    for exp in experiments:
        exp_name = exp['name']
        agg = all_experiment_results[exp_name]['aggregated']
        
        wave_ga_ba = agg['Wave-GA']['avg_minf1']
        wave_ga_std = agg['Wave-GA']['avg_minf1_std']
        
        # Find best baseline
        best_ba = 0.0
        best_name = ""
        for model_name, data in agg.items():
            if model_name != 'Wave-GA':
                if data['avg_minf1'] > best_ba:
                    best_ba = data['avg_minf1']
                    best_name = model_name
        
        wave_str = f"{wave_ga_ba:.4f}±{wave_ga_std:.4f}"
        best_str = f"{best_ba:.4f}"
        
        print(f"{exp_name:<35} {wave_str:<20} {best_name:<20} {best_str:<15}") 
    print("="*100)
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)

    print("="*80)