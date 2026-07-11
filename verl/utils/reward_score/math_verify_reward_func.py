
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(model_output: str, ground_truth, timeout_score: float = 0, return_details=True) -> bool:
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get('target', ground_truth)
    else:
        ground_truth = ground_truth
    
    format_correct = float(model_output.count("\\boxed") >= 1)
    ret_score = 0.
    extracted_contents = None

    if format_correct == 1:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        try:
            ret_score, extracted_contents = verify_func([ground_truth_boxed], [model_output])
        except Exception:
            pass
        except TimeoutException:
            ret_score = timeout_score
        if extracted_contents is not None and extracted_contents[1] == []:
            format_correct = 0.

    result = {"format_score": format_correct, "answer_score": ret_score, "score": ret_score}
    return result if return_details else result["score"]


if __name__ == "__main__":
    model_output = """### ✅ Final Answer:

$$
\\boxed{A = \\left\{ -4, \\frac{1}{2} \\right\}}
$$

"""
    ground_truth = {"target": "\\{0.5, -4\\}"}
    print(compute_score(model_output, ground_truth))
