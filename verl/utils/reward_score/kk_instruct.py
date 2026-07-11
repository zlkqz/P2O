import re
from typing import Dict, Tuple, Optional, List

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """

    answer_pattern = r'<my_answer>(.*?)</my_answer>'
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, solution_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, solution_str

def parse_solution_text_format(solution_text: List[str]) -> Dict[str, str]:
    """Parses ground truth solution text into status dictionary.
    
    Args:
        solution_text: Formatted solution text from dataset
        
    Returns:
        Dictionary mapping character names to their roles (knight/knave)
    """
    status_dict = {}
    print("\n[Ground Truth Parsing]")

    assert isinstance(solution_text, list), "Expected solution_text to be a list"

    solution_text = solution_text[0]
    
    for line in solution_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        match = re.search(r'\b([A-Za-z]+)\b.*?\b(knight|knave)\b', line, re.IGNORECASE)
        if match:
            name, role = match.groups()
            status_dict[name] = role.lower()
            print(f"  Found: {name} → {role}")
        else:
            print(f"  [Warning] Unparseable line: '{line}'")
    
    return status_dict

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

def validate_response_structure(processed_str: str, verbose=False):
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

def compute_score(solution_str: str, 
                 ground_truth: Dict[str, str],
                 format_score: int = 0,
                 answer_reward: float = 1.0,
                 return_details: bool = True) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: Dictionary containing ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        return_details: Whether to return detailed scores breakdown
        
    Returns:
        If return_details=False: Total score (sum of format and answer rewards)
        If return_details=True: Dict with 'total_score', 'format_score', 'answer_score'
    """
    print("\n" + "="*80)
    print(" Processing New Sample ".center(80, '='))
    
    solution_text = ground_truth.get('target', '')
    gt_status = parse_solution_text_format(solution_text)
    expected_names = list(gt_status.keys())
    print(f"[Ground Truth] Final identities: {gt_status}")

    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed_str}")

    format_score = 0.

    answer_score = 0
    if answer_text:
        pred_status = parse_model_answer(answer_text, expected_names)
        if pred_status:
            print(f"\n[Content Validation]")
            print(f"  Expected: {gt_status}")
            print(f"  Predicted: {pred_status}")
            
            if pred_status == gt_status:
                answer_score = 2
                print("  Content validation: FULL MATCH")
            else:
                answer_score = -2
                for name, role in pred_status.items():
                    if gt_status[name] == role:
                        answer_score = -1.5
                        print("  Content validation: PARTLY MATCH")
                        break
                if answer_score == -2:
                    print("  Content validation: MISMATCH")
        else:
            answer_score = -2
            print( "Fail to parse answer")
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    if return_details:
        return {
            'score': total_score,
            'format_score': format_score,
            'answer_score': answer_score
        }
    else:
        return total_score
    
if __name__ == "__main__":
    test_cases_pattern1 = [
        {
            "name": "Pattern 1 - Valid",
            "input": "<think>This is thinking</think><my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation here</write>",
            "expected": (True, 1)
        },
        {
            "name": "Pattern 1 - Valid with content",
            "input": "<think>Let me analyze the puzzle.\nA says B is a knave.</think><my_answer>A is a knight\nB is a knave</my_answer><golden_answer>A is a knight\nB is a knave</golden_answer><write>Since A is telling the truth...</write>",
            "expected": (True, 1)
        },
        {
            "name": "Pattern 1 - Missing think tag",
            "input": "<my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (False, None)
        },
        {
            "name": "Pattern 1 - Wrong order",
            "input": "<my_answer>A is knight</my_answer><think>Thinking</think><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (False, None)
        },
        {
            "name": "Pattern 1 - Extra tags",
            "input": "<think>This is thinking</think><search>extra</search><my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (False, None)
        },
        {
            "name": "Pattern 1 - Unclosed tag",
            "input": "<think>This is thinking<my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (False, None)
        }
    ]
    
    test_cases_pattern2 = [
        {
            "name": "Pattern 2 - Valid",
            "input": "<think>Initial thinking</think><search>Query here</search><information>Found info</information><my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (True, 2)
        },
        {
            "name": "Pattern 2 - Valid with complex content",
            "input": "<think>Let me search for more info</think><search>knights and knaves puzzles</search><information>Knights always tell truth, knaves always lie</information><my_answer>A is a knight\nB is a knave</my_answer><golden_answer>A is a knight\nB is a knave</golden_answer><write>Based on the information...</write>",
            "expected": (True, 2)
        },
        {
            "name": "Pattern 2 - Missing search tag",
            "input": "<think>Initial thinking</think><information>Found info</information><my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (False, None)
        },
        {
            "name": "Pattern 2 - Wrong tag order",
            "input": "<think>Initial thinking</think><information>Found info</information><search>Query here</search><my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (False, None)
        }
    ]
    
    test_cases_pattern3 = [
        {
            "name": "Pattern 3 - Valid",
            "input": "<think>Initial</think><search>Query</search><information>Info</information><my_answer>Answer</my_answer><golden_answer>Answer</golden_answer><think>Second think</think><write>Write</write>",
            "expected": (True, 3)
        },
        {
            "name": "Pattern 3 - Valid with content",
            "input": "<think>Let me search</think><search>puzzle logic</search><information>Found useful info</information><my_answer>A is a knight</my_answer><golden_answer>A is a knight</golden_answer><think>Let me reconsider</think><write>Final explanation</write>",
            "expected": (True, 3)
        }
    ]
    
    test_cases_pattern4 = [
        {
            "name": "Pattern 4 - Valid",
            "input": "<think>First</think><search>Q</search><information>I</information><think>Second</think><my_answer>A</my_answer><golden_answer>A</golden_answer><write>W</write>",
            "expected": (True, 4)
        },
        {
            "name": "Pattern 4 - Valid with newlines",
            "input": "<think>First think\nMultiline</think><search>search query</search><information>Found\ninfo</information><think>Second think</think><my_answer>A is knight</my_answer><golden_answer>A is knight</golden_answer><write>Explanation</write>",
            "expected": (True, 4)
        },
        {
            "name": "Pattern 4 - Think tags not adjacent",
            "input": "<think>First</think><think>Second</think><search>Q</search><information>I</information><my_answer>A</my_answer><golden_answer>A</golden_answer><write>W</write>",
            "expected": (False, None)
        }
    ]
    
    test_cases_pattern5 = [
        {
            "name": "Pattern 5 - Valid",
            "input": "<think>1</think><search>S</search><information>I</information><think>2</think><my_answer>A</my_answer><golden_answer>G</golden_answer><think>3</think><write>W</write>",
            "expected": (True, 5)
        },
        {
            "name": "Pattern 5 - Valid full content",
            "input": "<think>Initial analysis</think><search>logic puzzle rules</search><information>Knights tell truth</information><think>Based on info</think><my_answer>A is a knight\nB is a knave</my_answer><golden_answer>A is a knight\nB is a knave</golden_answer><think>Final check</think><write>Complete explanation</write>",
            "expected": (True, 5)
        },
        {
            "name": "Pattern 5 - Extra think tag",
            "input": "<think>1</think><search>S</search><information>I</information><think>2</think><my_answer>A</my_answer><golden_answer>G</golden_answer><think>3</think><think>4</think><write>W</write>",
            "expected": (False, None)
        }
    ]
    
    edge_cases = [
        {
            "name": "Empty string",
            "input": "",
            "expected": (False, None)
        },
        {
            "name": "Only opening tags",
            "input": "<think><my_answer><golden_answer><write>",
            "expected": (False, None)
        },
        {
            "name": "Only closing tags", 
            "input": "</think></my_answer></golden_answer></write>",
            "expected": (False, None)
        },
        {
            "name": "Nested tags (invalid)",
            "input": "<think>Text <think>nested</think> text</think><my_answer>A</my_answer><golden_answer>A</golden_answer><write>W</write>",
            "expected": (False, None)
        },
        {
            "name": "Tags with attributes (should fail)",
            "input": '<think id="1">Text</think><my_answer>A</my_answer><golden_answer>A</golden_answer><write>W</write>',
            "expected": (False, None)
        },
        {
            "name": "Mixed valid patterns (should fail)",
            "input": "<think>1</think><search>S</search><my_answer>A</my_answer><golden_answer>G</golden_answer><write>W</write>",
            "expected": (False, None)
        }
    ]
    
    all_test_cases = [
        ("Pattern 1", test_cases_pattern1),
        ("Pattern 2", test_cases_pattern2),
        ("Pattern 3", test_cases_pattern3),
        ("Pattern 4", test_cases_pattern4),
        ("Pattern 5", test_cases_pattern5),
        ("Edge Cases", edge_cases)
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for group_name, test_cases in all_test_cases:
        for test_case in test_cases:
            total_tests += 1
            print(f"\n{'='*60}")
            print(f"Test: {test_case['name']}")
            print(f"Input: {test_case['input'][:100]}..." if len(test_case['input']) > 100 else f"Input: {test_case['input']}")
            
            result = validate_response_structure(test_case['input'], verbose=False)
            
            if result == test_case['expected']:
                print(f"✅ PASSED - Expected: {test_case['expected']}, Got: {result}")
                passed_tests += 1
            else:
                print(f"❌ FAILED - Expected: {test_case['expected']}, Got: {result}")
                print("\nDebug output:")
                validate_response_structure(test_case['input'], verbose=True)
    
    print("\n" + "="*80)
    print(" Test Summary ".center(80, '='))
    print("="*80)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
    
    print("\n\n" + "="*60)
    print(" Special Character Tests ".center(60, '='))
    print("="*60)
    
    special_tests = [
        {
            "name": "Pattern 1 with special chars",
            "input": "<think>思考中文字符和emoji😊</think><my_answer>A is knight & B is knave</my_answer><golden_answer>A is knight & B is knave</golden_answer><write>解释说明</write>",
            "expected": (True, 1)
        },
        {
            "name": "Pattern 1 with XML entities",
            "input": "<think>A &gt; B</think><my_answer>A &amp; B</my_answer><golden_answer>A &amp; B</golden_answer><write>&lt;explanation&gt;</write>",
            "expected": (True, 1)
        }
    ]
    
    for test_case in special_tests:
        total_tests += 1
        print(f"\n{'='*60}")
        print(f"Test: {test_case['name']}")
        result = validate_response_structure(test_case['input'], verbose=False)
        if result == test_case['expected']:
            print(f"✅ PASSED")
            passed_tests += 1
        else:
            print(f"❌ FAILED - Expected: {test_case['expected']}, Got: {result}")