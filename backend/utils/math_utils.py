import math
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)

def normalize_likert_response(response: int, scale_min: int = 1, scale_max: int = 5) -> float:
    if not scale_min <= response <= scale_max:
        raise ValueError(f"Response {response} must be between {scale_min} and {scale_max}")
    
    return (response - scale_min) / (scale_max - scale_min)

def denormalize_to_likert(normalized_value: float, scale_min: int = 1, scale_max: int = 5) -> int:
    if not 0.0 <= normalized_value <= 1.0:
        raise ValueError(f"Normalized value {normalized_value} must be between 0.0 and 1.0")
    
    return round(normalized_value * (scale_max - scale_min) + scale_min)

def cosine_similarity(vector_a: Union[List[float], Dict[str, float]], 
                     vector_b: Union[List[float], Dict[str, float]]) -> float:
    # Convert dict vectors to lists with aligned keys
    if isinstance(vector_a, dict) and isinstance(vector_b, dict):
        common_keys = set(vector_a.keys()) & set(vector_b.keys())
        if not common_keys:
            raise ValueError("Vectors have no common dimensions")
        
        # Sort keys for consistent ordering
        sorted_keys = sorted(common_keys)
        vector_a = [vector_a[key] for key in sorted_keys]
        vector_b = [vector_b[key] for key in sorted_keys]
    
    # Convert to numpy arrays for efficient computation
    vec_a = np.array(vector_a)
    vec_b = np.array(vector_b)
    
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector shapes don't match: {vec_a.shape} vs {vec_b.shape}")
    
    # Calculate dot product and magnitudes
    dot_product = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    
    # Handle zero vectors
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    similarity = dot_product / (magnitude_a * magnitude_b)
    
    # Ensure result is in valid range due to floating point precision
    return max(-1.0, min(1.0, similarity))

def euclidean_distance(vector_a: Union[List[float], Dict[str, float]], 
                      vector_b: Union[List[float], Dict[str, float]]) -> float:

    # Convert dict vectors to lists with aligned keys
    if isinstance(vector_a, dict) and isinstance(vector_b, dict):
        common_keys = set(vector_a.keys()) & set(vector_b.keys())
        if not common_keys:
            raise ValueError("Vectors have no common dimensions")
        
        sorted_keys = sorted(common_keys)
        vector_a = [vector_a[key] for key in sorted_keys]
        vector_b = [vector_b[key] for key in sorted_keys]
    
    vec_a = np.array(vector_a)
    vec_b = np.array(vector_b)
    
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Vector shapes don't match: {vec_a.shape} vs {vec_b.shape}")
    
    return float(np.linalg.norm(vec_a - vec_b))

def entropy(probabilities: Union[List[float], Dict[str, float]], base: float = 2.0) -> float:
    if isinstance(probabilities, dict):
        probs = list(probabilities.values())
    else:
        probs = list(probabilities)
    
    # Validate probabilities
    probs = [p for p in probs if p > 1e-10]  # Filter out near-zero values
    
    if not probs:
        return 0.0
    
    total = sum(probs)
    if not 0.99 <= total <= 1.01:  # Allow small floating point errors
        logger.warning(f"Probabilities sum to {total}, not 1.0")
    
    # Calculate entropy
    entropy_value = 0.0
    for p in probs:
        if p > 0:
            entropy_value -= p * math.log(p, base)
    
    return entropy_value

def softmax(scores: Union[List[float], Dict[str, float]], 
           temperature: float = 1.0) -> Union[List[float], Dict[str, float]]:

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")
    
    is_dict = isinstance(scores, dict)
    if is_dict:
        keys = list(scores.keys())
        values = list(scores.values())
    else:
        values = list(scores)
    
    if not values:
        raise ValueError("Empty scores provided")
    
    # Apply temperature scaling
    scaled_values = [v / temperature for v in values]
    
    # Numerical stability: subtract max value
    max_value = max(scaled_values)
    exp_values = [math.exp(v - max_value) for v in scaled_values]
    
    # Normalize
    total = sum(exp_values)
    if total == 0:
        # Uniform distribution as fallback
        probabilities = [1.0 / len(exp_values)] * len(exp_values)
    else:
        probabilities = [v / total for v in exp_values]
    
    # Return same type as input
    if is_dict:
        return dict(zip(keys, probabilities))
    else:
        return probabilities

def kl_divergence(p: Union[List[float], Dict[str, float]], 
                 q: Union[List[float], Dict[str, float]]) -> float:

    # Convert to aligned lists if dicts
    if isinstance(p, dict) and isinstance(q, dict):
        common_keys = set(p.keys()) & set(q.keys())
        if not common_keys:
            raise ValueError("Distributions have no common dimensions")
        
        sorted_keys = sorted(common_keys)
        p_values = [p[key] for key in sorted_keys]
        q_values = [q[key] for key in sorted_keys]
    else:
        p_values = list(p)
        q_values = list(q)
    
    if len(p_values) != len(q_values):
        raise ValueError(f"Distribution lengths don't match: {len(p_values)} vs {len(q_values)}")
    
    kl_div = 0.0
    for p_i, q_i in zip(p_values, q_values):
        if p_i > 1e-10:  # Only consider non-zero probabilities in P
            if q_i <= 1e-10:  # Q is zero where P is non-zero
                return float('inf')  # KL divergence is infinite
            kl_div += p_i * math.log(p_i / q_i)
    
    return kl_div

def calculate_information_gain(current_entropy: float, 
                              weighted_entropies: List[Tuple[float, float]]) -> float:

    if not weighted_entropies:
        return 0.0
    
    # Calculate weighted average entropy after split
    total_weight = sum(weight for weight, _ in weighted_entropies)
    if total_weight == 0:
        return 0.0
    
    weighted_avg_entropy = sum(
        weight * entropy_val for weight, entropy_val in weighted_entropies
    ) / total_weight
    
    # Information gain = current entropy - weighted average entropy
    gain = current_entropy - weighted_avg_entropy
    return max(0.0, gain)  # Ensure non-negative

def weighted_average(values: List[float], weights: List[float]) -> float:

    if len(values) != len(weights):
        raise ValueError(f"Values and weights length mismatch: {len(values)} vs {len(weights)}")
    
    if not values:
        raise ValueError("Empty values provided")
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Weights sum to zero")
    
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    return weighted_sum / total_weight

def normalize_probabilities(probabilities: Union[List[float], Dict[str, float]]) -> Union[List[float], Dict[str, float]]:

    is_dict = isinstance(probabilities, dict)
    if is_dict:
        keys = list(probabilities.keys())
        values = list(probabilities.values())
    else:
        values = list(probabilities)
    
    total = sum(values)
    if total == 0:
        # Uniform distribution
        normalized = [1.0 / len(values)] * len(values)
    else:
        normalized = [v / total for v in values]
    
    if is_dict:
        return dict(zip(keys, normalized))
    else:
        return normalized

def calculate_confidence_interval(values: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:

    if not values:
        raise ValueError("Empty values list")
    
    if not 0 < confidence_level < 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
    
    values_array = np.array(values)
    mean = np.mean(values_array)
    std = np.std(values_array, ddof=1)  # Sample standard deviation
    
    # For small samples, use t-distribution; for large samples, normal approximation
    n = len(values)
    if n < 30:
        from scipy import stats
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
        margin_error = t_value * std / math.sqrt(n)
    else:
        # Normal approximation
        z_value = 1.96 if confidence_level == 0.95 else stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_value * std / math.sqrt(n)
    
    return (mean - margin_error, mean + margin_error)
