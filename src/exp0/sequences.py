"""Sequence formats A-E for 3SUM instances."""

import random
import string
from typing import List, Optional

from exp0.task3sum import Instance3Sum


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
    """Format A: Parallelizable CoT.

    Lexicographic order of all pairs (i, j) with i < j.
    For pair (i, j), if (x_i + x_j + x_k) % mod == 0 for some k distinct from i, j:
       write label(i) label(j) label(k) [or reduced: label(i)/label(j) + label(k)]
    Else write pair intermediate token / sum coordinate.
    """
    if rng is None:
        rng = random.Random()

    n = len(instance.tuples)
    d = len(instance.tuples[0])
    labels = get_token_labels(n)
    prefix = format_inputs(instance)

    # Map from tuple value -> list of indices
    val_to_indices = {}
    for idx, tup in enumerate(instance.tuples):
        val_to_indices.setdefault(tup, []).append(idx)

    cot_tokens = []

    for i in range(n):
        for j in range(i + 1, n):
            # Compute sum of i and j
            sum_ij = tuple((instance.tuples[i][dim] + instance.tuples[j][dim]) % mod for dim in range(d))
            target_k_val = tuple((-sum_ij[dim]) % mod for dim in range(d))

            # Check if target_k_val exists at any k distinct from i and j
            matching_k = None
            if target_k_val in val_to_indices:
                for k in val_to_indices[target_k_val]:
                    if k != i and k != j:
                        matching_k = k
                        break

            if vocab_reduction:
                # Vocab reduction: randomly pick one coordinate digit from sum_ij and single label char
                first_char = labels[i]
                second_char = labels[j]
                pair_label = first_char if rng.random() < 0.5 else second_char
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
    # For each digit dim, check 1-D 3sum
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


def format_e_neutral(instance: Instance3Sum, num_filler: Optional[int] = None, neutral_token: str = "#") -> str:
    """Format E: Neutral token arm (replaces '.' with neutral_token, e.g. '#')."""
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
