
import re
import string
import random
from typing import Dict, List, Optional

def extract_all_tags(text: str) -> List[str]:
    """Extract all XML-style tags from the text in order of appearance."""
    tag_pattern = r'</?[a-zA-Z_][\w\-\.]*>'
    tags = re.findall(tag_pattern, text)
    return tags

def validate_response_structure(processed_str: str, verbose=False):
    """Improved validation with strict pattern matching."""
    if verbose:
        print("\n[Structure Validation]")
    
    all_tags = extract_all_tags(processed_str)
    
    if verbose and len(all_tags) <= 20:
        print(f"  Found tags: {all_tags}")
    
    valid_sequences = [
        ['<think>', '</think>', '<my_answer>', '</my_answer>'],
    ]
    
    for pattern_idx, valid_sequence in enumerate(valid_sequences, 1):
        if all_tags == valid_sequence:
            if verbose:
                print(f"  Pattern {pattern_idx} matched successfully!")
            return True, pattern_idx
    
    if verbose:
        print("  [Error] No valid pattern matched")
        if len(all_tags) > 0:
            for pattern_idx, valid_sequence in enumerate(valid_sequences, 1):
                if set(all_tags) == set(valid_sequence) and len(all_tags) == len(valid_sequence):
                    print(f"  Note: Has same tags as Pattern {pattern_idx} but in wrong order")
                elif set(all_tags).issubset(set(valid_sequence)):
                    print(f"  Note: Missing tags from Pattern {pattern_idx}")
    
    return False, None


def extract_solution(solution_str):
    """Extract the equation from the solution string."""

    answer_pattern = r'<my_answer>(.*?)</my_answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    matches = [m for m in matches if m.group(1).strip() != "and"]

    if len(matches) == 0:
        return None
    
    return matches[0].group(1).strip()


def extract_winner(text):
    text = text.strip()
    normalized_text = re.sub(r"\s+", " ", text)
    
    pattern = r"^\[\[\s*([A-Za-z])\s*([<>])\s*([A-Za-z])\s*\]\]$"
    match = re.match(pattern, normalized_text)
    if not match:
        return None

    left_symbol = match.group(1).upper()
    operator_symbol = match.group(2)
    right_symbol = match.group(3).upper()

    if left_symbol not in {"A", "B"} or right_symbol not in {"A", "B"}:
        return None

    if operator_symbol == "<":
        return right_symbol.strip()
    elif operator_symbol == ">":
        return left_symbol.strip()
    else:
        return None


def compute_score(solution_str, ground_truth, format_score=1., score=1., return_details: bool = True):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """

    format_correct, _ = validate_response_structure(solution_str)
    format_score = format_score if format_correct else 0

    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"+"*30)
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
        
    
    if answer is None:
        if return_details:
            return {
                'score': 0.,
                'format_score': 0.,
                'answer_score': 0
            }
        else:
            return 0.
    else:
        answer_winner = extract_winner(answer)
        target_winner = extract_winner(ground_truth["target"])
        if answer_winner is not None and target_winner is not None and answer_winner == target_winner:
            answer_score = 1.
        else:
            answer_score = 0.
        answer_score = answer_score * score
        if answer_winner is None:
            format_score=  0.
        total_score = answer_score + format_score

        if do_print:
            print(f"Format score: {format_score}")
            print(f"Answer score: {answer_score}")
            print(f"+"*30)

        if return_details:
            return {
                'score': total_score,
                'format_score': format_score,
                'answer_score': answer_score
            }
        else:
            return total_score