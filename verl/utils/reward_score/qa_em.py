
import re
import string
import random
from typing import Dict, List, Optional

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score

def parse_model_answer(answer_text: str, expected_names: list) -> Optional[Dict[str, str]]:
    """Parses model's answer text into status dictionary.
    
    Args:
        answer_text: Text extracted from model's <answer> tags
        expected_names: List of character names requiring identification
        
    Returns:
        Dictionary mapping character names to predicted roles, or None if incomplete
    """
    status_dict = {}
    print("\n[Model Answer Parsing]")
    print(f"  Expected characters: {expected_names}")

    knight_count = answer_text.lower().count('knight')
    knave_count = answer_text.lower().count('knave')

    print(f"  Number of predicted roles: {knight_count + knave_count}")
    if knight_count + knave_count != len(expected_names):
        print(f"  [Error] Number of characters mismatch: {knight_count + knave_count} != {len(expected_names)}")
        return None

    for name in expected_names:
        pattern = re.compile(
            rf'\b{re.escape(name)}\b\s+is\s+a\s+\b(knight|knave)\b', 
            re.IGNORECASE
        )
        match = pattern.search(answer_text)
        
        if match:
            role = match.group(1).lower()
            status_dict[name] = role
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Error] Missing identification for {name}")
            return None
    
    return status_dict

def extract_all_tags(text: str) -> List[str]:
    """Extract all XML-style tags from the text in order of appearance."""
    tag_pattern = r'</?[a-zA-Z_][\w\-\.]*>'
    tags = re.findall(tag_pattern, text)
    return tags

def validate_response_structure(processed_str: str, verbose=True):
    """Improved validation with strict pattern matching."""
    if verbose:
        print("\n[Structure Validation]")
    
    all_tags = extract_all_tags(processed_str)
    
    if verbose and len(all_tags) <= 20:
        print(f"  Found tags: {all_tags}")
    
    valid_sequences = [
        ['<think>', '</think>', '<my_answer>', '</my_answer>', 
         '<golden_answer>', '</golden_answer>', '<write>', '</write>'],
        
        ['<think>', '</think>', '<search>', '</search>', 
         '<information>', '</information>', '<my_answer>', '</my_answer>', 
         '<golden_answer>', '</golden_answer>', '<write>', '</write>'],
        
        ['<think>', '</think>', '<search>', '</search>', 
         '<information>', '</information>', '<my_answer>', '</my_answer>', 
         '<golden_answer>', '</golden_answer>', '<think>', '</think>', 
         '<write>', '</write>'],
        
        ['<think>', '</think>', '<search>', '</search>', 
         '<information>', '</information>', '<think>', '</think>', 
         '<my_answer>', '</my_answer>', '<golden_answer>', '</golden_answer>', 
         '<write>', '</write>'],
        
        ['<think>', '</think>', '<search>', '</search>', 
         '<information>', '</information>', '<think>', '</think>', 
         '<my_answer>', '</my_answer>', '<golden_answer>', '</golden_answer>', 
         '<think>', '</think>', '<write>', '</write>'],

        ['<search>', '</search>', '<information>', '</information>', 
         '<think>', '</think>', '<my_answer>', '</my_answer>', 
         '<golden_answer>', '</golden_answer>', '<write>', '</write>'],

        ['<search>', '</search>', '<information>', '</information>',
         '<think>', '</think>', '<my_answer>', '</my_answer>', 
         '<golden_answer>', '</golden_answer>', '<think>', '</think>',
         '<write>', '</write>']

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


def compute_score_em(solution_str, ground_truth, method='strict', format_score=1., score=1., return_details: bool = True):
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
                'score': format_score,
                'format_score': format_score,
                'answer_score': 0
            }
        else:
            return format_score
    else:
        answer_score = em_check(answer, ground_truth['target'])
        answer_score = answer_score * score
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
