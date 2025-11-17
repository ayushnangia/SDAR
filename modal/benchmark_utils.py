"""
Utility functions for GSM8K and AIME benchmarking with SDAR models.
"""
import re
from typing import Optional, Dict, Any, List


def extract_answer_gsm8k(text: str, method: str = "strict") -> Optional[str]:
    """
    Extract numerical answer from GSM8K generated text.

    Args:
        text: Generated text from the model
        method: "strict" uses #### marker, "flexible" finds last number

    Returns:
        Extracted answer as string, or None if not found
    """
    # First, try to extract from \boxed{} format (preferred)
    # Handle typos like \boxedboxed{} or \boxxed{}
    boxed_patterns = [r'\boxed{', r'\boxedboxed{', r'\boxxed{']
    boxed_start = -1
    pattern_len = 0

    for pattern in boxed_patterns:
        idx = text.find(pattern)
        if idx != -1:
            boxed_start = idx
            pattern_len = len(pattern)
            break

    if boxed_start != -1:
        # Find matching closing brace
        start_pos = boxed_start + pattern_len
        brace_count = 1
        pos = start_pos
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            boxed_content = text[start_pos:pos-1].strip()
            # If it contains LaTeX, try to extract just the number
            # Remove \frac{a}{b} and similar
            if r'\frac{' in boxed_content:
                # Extract numerator from fraction as fallback
                frac_match = re.search(r'\\frac\{([^}]+)\}', boxed_content)
                if frac_match:
                    return frac_match.group(1).strip()
            # Clean up LaTeX commands and formatting
            boxed_content = re.sub(r'\\[a-zA-Z]+', '', boxed_content)  # Remove \text, \, etc
            boxed_content = boxed_content.replace('{', '').replace('}', '').strip()

            # Extract just the number from content with units or expressions
            # Look for: number with optional $ prefix, optional commas, optional decimal
            # Stop at spaces, letters, or operators (except negative sign at start)
            number_match = re.search(r'\$?\s*(-?[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?)', boxed_content)
            if number_match:
                return number_match.group(1).strip()
            elif boxed_content and not re.search(r'[a-zA-Z]{4,}', boxed_content):
                # Only return if it doesn't contain long words (like "vacuum cleaners")
                return boxed_content
            # If boxed content is garbage (all text), fall through to flexible extraction

    if method == "strict":
        # Look for the #### marker followed by a number
        pattern = r"####\s*([-]?[0-9,\.]+)"
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    elif method == "flexible":
        # Remove common LaTeX commands to avoid partial extractions
        cleaned_text = re.sub(r"\\frac\{[^}]*\}?", "", text)  # Remove \frac{} and incomplete \frac{
        cleaned_text = re.sub(r"\\[a-zA-Z]+\{[^}]*\}?", "", cleaned_text)  # Remove \cmd{} patterns
        cleaned_text = re.sub(r"\\%", "%", cleaned_text)  # Convert \% to %

        # Find the last number in the cleaned text (handles various formats)
        # Look for numbers: optional $, digits with optional commas, optional decimal part
        # Must have at least one digit before or after decimal point
        pattern = r"\$?\s*(-?[0-9]+(?:,[0-9]+)*(?:\.[0-9]+)?|-?[0-9]*\.[0-9]+)"
        matches = re.findall(pattern, cleaned_text)
        if matches:
            # Filter out matches that are just punctuation or too short
            valid_matches = [m.strip() for m in matches if m.strip() and not m.strip() in [',', '.', '-']]
            if valid_matches:
                # Return the last match, which is likely the final answer
                return valid_matches[-1]

    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize numerical answer for comparison.

    Args:
        answer: Answer string to normalize

    Returns:
        Normalized answer (no commas, dollar signs, whitespace)
    """
    if answer is None:
        return ""

    # Remove LaTeX backslashes and common formatting
    normalized = answer.replace("\\", "").replace(",", "").replace("$", "").replace("%", "").strip()

    # Handle decimal points - convert to float then back to string to standardize
    try:
        # Try to parse as float to handle cases like "3.0" vs "3"
        num = float(normalized)
        # If it's a whole number, return as int string
        if num.is_integer():
            return str(int(num))
        return str(num)
    except ValueError:
        # If not a valid number, return as-is
        return normalized


def evaluate_answer(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.

    Args:
        predicted: Predicted answer from model
        ground_truth: Ground truth answer from dataset

    Returns:
        True if answers match, False otherwise
    """
    # Extract ground truth answer (it has #### marker)
    gt_answer = extract_answer_gsm8k(ground_truth, method="strict")

    if gt_answer is None or predicted is None:
        return False

    # Normalize both answers
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(gt_answer)

    return pred_norm == gt_norm


def compute_accuracy(results: list[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute accuracy metrics from evaluation results.

    Args:
        results: List of dicts with 'correct' boolean field

    Returns:
        Dict with accuracy and count statistics
    """
    total = len(results)
    if total == 0:
        return {"accuracy": 0.0, "correct": 0, "total": 0}

    correct = sum(1 for r in results if r.get("correct", False))

    return {
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
        "percentage": f"{100 * correct / total:.2f}%"
    }


def format_prompt_gsm8k(question: str, include_instruction: bool = True) -> str:
    """
    Format GSM8K question into a prompt.

    Args:
        question: The math problem question
        include_instruction: Whether to add instruction for answer format

    Returns:
        Formatted prompt string
    """
    if include_instruction:
        return (
            f"{question}\n\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}"
        )
    return question


# ============================================================================
# AIME-specific functions
# ============================================================================

def extract_answer_aime(text: str) -> Optional[int]:
    """
    Extract integer answer (000-999) from AIME generated text.

    AIME answers must be integers in the range 000-999 (inclusive).

    Args:
        text: Generated text from the model

    Returns:
        Integer answer if valid (0-999), None otherwise
    """
    # First, try to extract from \boxed{} format (preferred for AIME)
    boxed_patterns = [r'\boxed{', r'\boxedboxed{', r'\boxxed{']
    boxed_start = -1
    pattern_len = 0

    for pattern in boxed_patterns:
        idx = text.find(pattern)
        if idx != -1:
            boxed_start = idx
            pattern_len = len(pattern)
            break

    if boxed_start != -1:
        # Find matching closing brace
        start_pos = boxed_start + pattern_len
        brace_count = 1
        pos = start_pos
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        if brace_count == 0:
            boxed_content = text[start_pos:pos-1].strip()

            # Clean up LaTeX commands
            boxed_content = re.sub(r'\\[a-zA-Z]+', '', boxed_content)
            boxed_content = boxed_content.replace('{', '').replace('}', '').strip()

            # Try to extract integer from boxed content
            number_match = re.search(r'(-?[0-9]+)', boxed_content)
            if number_match:
                try:
                    answer = int(number_match.group(1))
                    # Validate range 0-999
                    if 0 <= answer <= 999:
                        return answer
                except ValueError:
                    pass

    # Fallback: find last integer in text
    # Remove LaTeX commands first
    cleaned_text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}?', '', text)

    # Find all integers
    integers = re.findall(r'\b([0-9]+)\b', cleaned_text)

    # Try from last to first (most recent integer is likely the answer)
    for num_str in reversed(integers):
        try:
            answer = int(num_str)
            if 0 <= answer <= 999:
                return answer
        except ValueError:
            continue

    return None


def normalize_answer_aime(answer: Optional[int]) -> str:
    """
    Normalize AIME answer for comparison.

    AIME answers are integers 000-999, typically formatted without leading zeros
    for comparison purposes.

    Args:
        answer: Integer answer (or None)

    Returns:
        Normalized answer as string
    """
    if answer is None:
        return ""

    # Simply return as string (no leading zeros for comparison)
    return str(answer)


def evaluate_answer_aime(predicted: Optional[int], ground_truth: str) -> bool:
    """
    Check if predicted AIME answer matches ground truth.

    Args:
        predicted: Predicted integer answer from model
        ground_truth: Ground truth answer string from dataset

    Returns:
        True if answers match, False otherwise
    """
    if predicted is None:
        return False

    # Try to parse ground truth as integer
    try:
        # Ground truth might be a string like "042" or "42"
        gt_answer = int(ground_truth.strip())
    except ValueError:
        # If ground truth is not a simple integer, try to extract it
        gt_extracted = extract_answer_aime(ground_truth)
        if gt_extracted is None:
            return False
        gt_answer = gt_extracted

    # Compare integers directly
    return predicted == gt_answer


def format_prompt_aime(problem: str, include_instruction: bool = True) -> str:
    """
    Format AIME problem into a prompt.

    Args:
        problem: The AIME problem text (may contain LaTeX)
        include_instruction: Whether to add instruction for answer format

    Returns:
        Formatted prompt string
    """
    if include_instruction:
        return (
            f"{problem}\n\n"
            "Please reason step by step, and put your final answer "
            "(an integer from 000 to 999) within \\boxed{{}}"
        )
    return problem


def compute_pass_at_k(problem_results: List[Dict[str, Any]], k: int) -> float:
    """
    Compute pass@k metric for a set of problems.

    pass@k = proportion of problems where at least one of k attempts is correct

    Args:
        problem_results: List of dicts, each containing:
            - 'problem_id': identifier for the problem
            - 'attempts': list of dicts with 'correct' boolean field
        k: Number of attempts to consider (1, 8, 32, etc.)

    Returns:
        pass@k score (0.0 to 1.0)
    """
    if not problem_results:
        return 0.0

    passed = 0
    for problem in problem_results:
        attempts = problem.get('attempts', [])
        # Take first k attempts
        k_attempts = attempts[:k]
        # Problem passes if ANY attempt in first k is correct
        if any(attempt.get('correct', False) for attempt in k_attempts):
            passed += 1

    return passed / len(problem_results)


def compute_aime_metrics(problem_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute comprehensive AIME metrics including pass@1, pass@8, pass@32.

    Args:
        problem_results: List of problem results with multiple attempts per problem

    Returns:
        Dict with various metrics
    """
    if not problem_results:
        return {
            "pass@1": 0.0,
            "pass@8": 0.0,
            "pass@32": 0.0,
            "total_problems": 0,
            "total_attempts": 0
        }

    total_problems = len(problem_results)
    total_attempts = sum(len(p.get('attempts', [])) for p in problem_results)

    return {
        "pass@1": compute_pass_at_k(problem_results, 1),
        "pass@8": compute_pass_at_k(problem_results, 8),
        "pass@32": compute_pass_at_k(problem_results, 32),
        "total_problems": total_problems,
        "total_attempts": total_attempts,
        "pass@1_percentage": f"{100 * compute_pass_at_k(problem_results, 1):.2f}%",
        "pass@8_percentage": f"{100 * compute_pass_at_k(problem_results, 8):.2f}%",
        "pass@32_percentage": f"{100 * compute_pass_at_k(problem_results, 32):.2f}%",
    }
