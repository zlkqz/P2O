
from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ["reward_bench_2_memory_search_first", "skywork_preference_memory_search_first", "reward_bench_2_memory_search_first_v2", "skywork_preference_memory_search_first_v2", "reward_bench_2_memory_search_first_v3", "skywork_preference_memory_search_first_v3"]:
        from . import rm
        res = rm.compute_score(solution_str, ground_truth)

    elif data_source in ["reward_bench_2_baseline", "skywork_preference_baseline", "reward_bench_2_baseline_v2", "skywork_preference_baseline_v2"]:
        from . import rm_instruct
        res = rm_instruct.compute_score(solution_str, ground_truth)

    elif data_source in ["skywork_preference_memory_search_first_with_memory_select", "reward_bench_2_memory_search_first_with_memory_select"]:
        from . import rm_with_memory_selection
        res = rm_with_memory_selection.compute_score(solution_str, ground_truth)

    elif data_source in ["DigitalLearningGmbH/MATH-lighteval_instruct"]:
        from . import math_instruct
        res = math_instruct.compute_score(solution_str, ground_truth)

    elif data_source == "kk_logic_instruct":

        from . import kk_instruct
        res = kk_instruct.compute_score(solution_str, ground_truth)

    elif data_source in ["nq_instruct", "2wikimultihopqa_instruct", "bamboogle_instruct", "hotpotqa_instruct", "musique_instruct", "popqa_instruct", "triviaqa_instruct"]:

        from . import qa_em_instruct
        res = qa_em_instruct.compute_score_em(solution_str, ground_truth)

    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward_func

        res = math_reward_func.compute_score(solution_str, ground_truth)


    elif data_source in ["openr1_math"]:
        from . import math_verify_reward_func
        res = math_verify_reward_func.compute_score(solution_str, ground_truth)
    
    elif "webins" in data_source:
        from . import webins
        verifier_base_url = extra_info.get("verifier_base_url", None)
        if verifier_base_url is None:
            raise ValueError(f"verifier_base_url is not found in extra_info: {extra_info}")
        question = extra_info.get("original_question", None)
        if question is None:
            raise ValueError(f"original_question is not found in extra_info: {extra_info}")
        res = webins.compute_score(solution_str, question, ground_truth, verifier_base_url=verifier_base_url)

    elif data_source == "deepscaler":
        from . import deepscaler
        res = deepscaler.compute_score(solution_str, ground_truth)

    elif data_source == "deepmath":
        from . import deepmath
        res = deepmath.compute_score(solution_str, ground_truth)
    
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        if sandbox_fusion_url:
            from . import sandbox_fusion

            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            from . import prime_code

            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    elif data_source == "kk_logic":

        from . import kk
        res = kk.compute_score(solution_str, ground_truth)

    elif data_source in ["nq", "2wikimultihopqa", "bamboogle", "hotpotqa", "musique", "popqa", "triviaqa"]:

        from . import qa_em
        res = qa_em.compute_score_em(solution_str, ground_truth)

    elif "preference" in data_source:

        from . import rm
        res = rm.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")


    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
