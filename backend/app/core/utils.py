import math
from typing import Dict, List, Union
import logging

logger = logging.getLogger(__name__)

def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """
    Calculate cosine similarity between two trait vectors
    
    Args:
        vec_a: First vector (user traits)
        vec_b: Second vector (department traits)
        
    Returns:
        Similarity score between 0 and 1
    """
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not common_keys:
        return 0.0
    
    # Calculate dot product and magnitudes
    dot_product = sum(vec_a[k] * vec_b[k] for k in common_keys)
    mag_a = math.sqrt(sum(vec_a[k]**2 for k in common_keys))
    mag_b = math.sqrt(sum(vec_b[k]**2 for k in common_keys))
    
    # Handle zero vectors
    if mag_a == 0 or mag_b == 0:
        return 0.0
    
    similarity = dot_product / (mag_a * mag_b)
    return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

def softmax(scores: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:
    """
    Convert scores to probabilities using softmax
    
    Args:
        scores: Dictionary of raw scores
        temperature: Temperature parameter for softmax
        
    Returns:
        Dictionary of probabilities that sum to 1.0
    """
    if not scores:
        return {}
    
    # Apply temperature scaling
    scaled = {k: v / temperature for k, v in scores.items()}
    
    # Numerical stability - subtract max
    max_val = max(scaled.values())
    exp_vals = {k: math.exp(v - max_val) for k, v in scaled.items()}
    
    # Normalize
    total = sum(exp_vals.values())
    if total == 0:
        # Uniform distribution fallback
        return {k: 1.0 / len(scores) for k in scores.keys()}
    
    return {k: v / total for k, v in exp_vals.items()}

def calculate_entropy(probabilities: Dict[str, float]) -> float:
    """
    Calculate Shannon entropy of probability distribution
    
    Args:
        probabilities: Dictionary of probabilities
        
    Returns:
        Entropy value (higher = more uncertain)
    """
    entropy = 0.0
    for prob in probabilities.values():
        if prob > 1e-10:  # Avoid log(0)
            entropy -= prob * math.log2(prob)
    return entropy

def normalize_probabilities(probs: Dict[str, float]) -> Dict[str, float]:
    """Ensure probabilities sum to 1.0"""
    total = sum(probs.values())
    if total == 0:
        return {k: 1.0 / len(probs) for k in probs.keys()}
    return {k: v / total for k, v in probs.items()}

def calculate_information_gain(current_entropy: float, 
                              weighted_entropies: List[tuple]) -> float:
    """
    Calculate information gain from entropy reduction
    
    Args:
        current_entropy: Current entropy before question
        weighted_entropies: List of (weight, entropy) tuples for possible outcomes
        
    Returns:
        Information gain (0.0 to current_entropy)
    """
    if not weighted_entropies:
        return 0.0
    
    total_weight = sum(weight for weight, _ in weighted_entropies)
    if total_weight == 0:
        return 0.0
    
    # Calculate weighted average entropy after split
    weighted_avg = sum(weight * entropy for weight, entropy in weighted_entropies) / total_weight
    
    # Information gain = current entropy - expected entropy
    gain = current_entropy - weighted_avg
    return max(0.0, gain)

def normalize_likert_response(response: int, scale_min: int = 1, scale_max: int = 5) -> float:
    """Convert Likert response to normalized 0-1 scale"""
    if not scale_min <= response <= scale_max:
        raise ValueError(f"Response {response} must be between {scale_min} and {scale_max}")
    return (response - scale_min) / (scale_max - scale_min)

def get_confidence_level(probability: float) -> str:
    """Convert probability to confidence level string"""
    if probability >= 0.85:
        return "high"
    elif probability >= 0.70:
        return "medium"
    else:
        return "low"

# Standard trait names for consistency
TRAIT_NAMES = [
    "coding_aptitude", "hardware_technical", "digital_design", "web_development",
    "visual_creativity", "content_creation", "hands_on_crafting", "innovation_ideation", 
    "stakeholder_management", "team_collaboration", "public_interaction", "networking_ability",
    "logistics_coordination", "process_management", "event_execution", "strategic_planning",
    "business_development", "financial_management", "leadership_initiative", "analytical_thinking"
]