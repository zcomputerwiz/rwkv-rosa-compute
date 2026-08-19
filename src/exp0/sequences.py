"""Sequence formats A-E for 3SUM instances."""

import random
import string
from typing import List, Optional

from exp0.task3sum import Instance3Sum, matching_k_after_pair


def get_token_labels(n: int) -> List[str]:
    """Get uppercase letter labels for n tuple positions: A, B, C, ... Z, AA, AB, ..."""
    labels = []
    for i in range(n):
        if i < 26:
            labels.append(string.ascii_uppercase[i])
        else:
            first = string.ascii_uppercase[(i // 26) - 1]
            second = string.ascii_uppercase[i % 26]
            labels.append(f"{first}{second}")
    return labels


def format_inputs(instance: Instance3Sum) -> str:
    """Format input tuples: e.g. A05 B75 C22 D13 : """
    labels = get_token_labels(len(instance.tuples))
    parts = []
    for label, tup in zip(labels, instance.tuples):
        digits_str = "".join(str(d) for d in tup)
        parts.append(f"{label}{digits_str}")
    return " ".join(parts) + " : "


def format_a_parallel_cot(
    instance: Instance3Sum,
    mod: int = 10,
    vocab_reduction: bool = True,
    rng: Optional[random.Random] = None,
) -> str:
    """Format A: source-faithful parallelizable CoT.

    Pairs are enumerated lexicographically as ``i < j``. Matching third tuples
    are searched only in the suffix ``k > j``, matching the published Match-3
    dense solver. Thus a solution triple ``i < j < k`` is exposed exactly once,
    at pair ``(i, j)``, instead of redundantly through all three unordered pairs.
    """
    if rng is None:
        rng = random.Random()

    n = len(instance.tuples)
    labels = get_token_labels(n)
    prefix = format_inputs(instance)
    cot_tokens = []

    for i in range(n):
        for j in range(i + 1, n):
            sum_ij, matching_k = matching_k_after_pair(
                instance.tuples,
                i,
                j,
                mod=mod,
            )

            if vocab_reduction:
                pair_label = labels[i] if rng.random() < 0.5 else labels[j]
                sum_digit = str(rng.choice(sum_ij))
                if matching_k is not None:
                    cot_tokens.append(f"{pair_label} {labels[matching_k]}")
                else:
                    cot_tokens.append(f"{pair_label} {sum_digit}")
            else:
                pair_label = f"{labels[i]}{labels[j]}"
                sum_str = "".join(str(s) for s in sum_ij)
                if matching_k is not None:
                    cot_tokens.append(f"{pair_label} {labels[matching_k]}")
                else:
                    cot_tokens.append(f"{pair_label} {sum_str}")

    ans_str = f"ANS {instance.has_3sum}"
    return prefix + " ".join(cot_tokens) + " " + ans_str


def format_b_filler(instance: Instance3Sum, num_filler: Optional[int] = None) -> str:
    """Format B: Filler tokens replacing CoT. Default num_filler = n^2."""
    n = len(instance.tuples)
    if num_filler is None:
        num_filler = n * n
    prefix = format_inputs(instance)
    filler_str = " ".join(["."] * num_filler)
    ans_str = f"ANS {instance.has_3sum}"
    if num_filler > 0:
        return prefix + filler_str + " " + ans_str
    else:
        return prefix + ans_str


def format_c_immediate(instance: Instance3Sum) -> str:
    """Format C: Immediate answer (N=0 filler)."""
    return format_b_filler(instance, num_filler=0)


def format_d_serial_cot(instance: Instance3Sum, mod: int = 10) -> str:
    """Format D: Instance-adaptive (serial) CoT.

    Reduces d-dimensional 3SUM to 1-D checks sequentially digit by digit.
    """
    prefix = format_inputs(instance)
    n = len(instance.tuples)
    d = len(instance.tuples[0])
    labels = get_token_labels(n)

    cot_tokens = []
    for dim in range(d):
        cot_tokens.append(f"DIM {dim}")
        for i in range(n):
            for j in range(i + 1, n):
                s1d = (instance.tuples[i][dim] + instance.tuples[j][dim]) % mod
                cot_tokens.append(f"{labels[i]}{labels[j]}_{dim} {s1d}")
                for k in range(j + 1, n):
                    if (s1d + instance.tuples[k][dim]) % mod == 0:
                        cot_tokens.append(f"MATCH {labels[i]}{labels[j]}{labels[k]}_{dim}")

    ans_str = f"ANS {instance.has_3sum}"
    return prefix + " ".join(cot_tokens) + " " + ans_str


def format_e_neutral(
    instance: Instance3Sum,
    num_filler: Optional[int] = None,
    neutral_token: str = "#",
) -> str:
    """Format E: Neutral token arm (replaces '.' with neutral_token)."""
    n = len(instance.tuples)
    if num_filler is None:
        num_filler = n * n
    prefix = format_inputs(instance)
    filler_str = " ".join([neutral_token] * num_filler)
    ans_str = f"ANS {instance.has_3sum}"
    if num_filler > 0:
        return prefix + filler_str + " " + ans_str
    else:
        return prefix + ans_str
