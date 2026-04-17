"""
Needle-in-Haystack Controlled Study for Exit Layer vs Long-Range Dependency.

Constructs synthetic sequences where long-range dependency is *known by design*,
then checks whether the model's exit layer for "needle-dependent" tokens is higher
than for non-dependent tokens.

Design:
  1. Embed a "needle" fact (e.g., "The capital of Freedonia is Zanthar.")
     at a controlled distance from a "query" that requires it
     (e.g., "The capital of Freedonia is").
  2. Surround with filler text (haystack) from WikiText or repeated patterns.
  3. Measure exit_layer at the query's answer token position.
  4. Compare exit layers when the needle is present vs. absent (control),
     and when the needle distance is short vs. long.

This directly tests: "tokens that require long-range information need more layers."

Usage:
    python experiments/needle_in_haystack.py --model Qwen/Qwen3-1.7B
    python experiments/needle_in_haystack.py --model Qwen/Qwen3-1.7B --n_trials 10  # quick test
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NeedleTrial:
    """One trial: a sequence with a needle placed at a known position."""
    name: str                    # e.g. "capital_freedonia"
    needle_text: str             # the fact to embed
    query_text: str              # prompt leading up to the answer
    answer_text: str             # the expected completion
    needle_distance: int         # distance in tokens between needle end and query start
    needle_present: bool         # False = control (needle replaced with filler)


@dataclass
class TrialResult:
    name: str
    needle_distance: int
    needle_present: bool
    answer_exit_layer: float     # mean exit layer over answer tokens
    answer_exit_layers: List[int]  # per-token exit layers for answer tokens
    answer_correct: bool         # did top-1 at final layer predict the answer?
    full_context_prob: float     # probability of answer token at final layer
    num_answer_tokens: int


# ---------------------------------------------------------------------------
# Needle-Haystack templates
# ---------------------------------------------------------------------------

NEEDLE_TEMPLATES = [
    {
        "name": "capital_freedonia",
        "needle": "The capital of Freedonia is Zanthar.",
        "query": "According to the information above, the capital of Freedonia is",
        "answer": " Zanthar",
    },
    {
        "name": "ceo_nexcorp",
        "needle": "The CEO of NexCorp is Dr. Helia Voss.",
        "query": "As stated earlier, the CEO of NexCorp is",
        "answer": " Dr",  # first token of multi-token answer
    },
    {
        "name": "population_keldara",
        "needle": "The population of Keldara is 47382.",
        "query": "The population of Keldara, as mentioned above, is",
        "answer": " 47",  # first token
    },
    {
        "name": "color_vexium",
        "needle": "The color of Vexium crystals is deep violet.",
        "query": "As noted previously, the color of Vexium crystals is",
        "answer": " deep",
    },
    {
        "name": "year_founded_arcanis",
        "needle": "The Arcanis Institute was founded in 1847.",
        "query": "According to the text, the Arcanis Institute was founded in",
        "answer": " 1847",
    },
]


def generate_filler(tokenizer, n_tokens: int) -> str:
    """Generate filler text of approximately n_tokens length."""
    # Use a repetitive but natural-looking filler
    filler_sentences = [
        "The weather was pleasant that afternoon.",
        "Several researchers gathered in the conference room to discuss the findings.",
        "The river flowed steadily through the valley below.",
        "Books lined the shelves from floor to ceiling in the old library.",
        "A gentle breeze carried the scent of flowers through the open window.",
        "The committee reviewed the proposal and made several recommendations.",
        "Mountains rose in the distance, their peaks covered with snow.",
        "Students attended lectures throughout the morning and early afternoon.",
        "The market was bustling with traders selling various goods.",
        "Ancient ruins dotted the landscape, remnants of a forgotten civilization.",
    ]
    # Build filler by cycling through sentences
    filler = ""
    idx = 0
    while True:
        candidate = filler + " " + filler_sentences[idx % len(filler_sentences)]
        if len(tokenizer.encode(candidate, add_special_tokens=False)) >= n_tokens:
            break
        filler = candidate
        idx += 1
    return filler.strip()


def build_sequence(
    tokenizer,
    template: dict,
    needle_distance: int,
    needle_present: bool,
    total_length: int = 2048,
) -> tuple:
    """
    Build a token sequence with needle at a controlled distance from query.

    Returns: (input_ids, answer_start_idx, answer_end_idx, trial_info)
    where answer_start_idx and answer_end_idx are token positions of the answer.
    """
    needle_text = template["needle"]
    query_text = template["query"]
    answer_text = template["answer"]

    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
    answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)

    # We need: [prefix_filler] [needle] [middle_filler of needle_distance tokens] [query] [answer] [suffix_filler]
    # needle_distance = number of tokens between needle end and query start

    middle_filler_text = generate_filler(tokenizer, needle_distance)
    middle_filler_tokens = tokenizer.encode(middle_filler_text, add_special_tokens=False)
    # Trim to exact distance
    middle_filler_tokens = middle_filler_tokens[:needle_distance]

    # If needle not present, replace needle with more filler
    if needle_present:
        needle_section = needle_tokens
    else:
        replacement_filler = generate_filler(tokenizer, len(needle_tokens) + 5)
        replacement_tokens = tokenizer.encode(replacement_filler, add_special_tokens=False)
        needle_section = replacement_tokens[:len(needle_tokens)]

    # Build: prefix + needle + middle_filler + query + answer
    # Calculate prefix length to hit total_length
    core_len = len(needle_section) + len(middle_filler_tokens) + len(query_tokens) + len(answer_tokens)
    prefix_len = max(32, total_length - core_len)  # at least 32 tokens prefix

    prefix_filler_text = generate_filler(tokenizer, prefix_len)
    prefix_tokens = tokenizer.encode(prefix_filler_text, add_special_tokens=False)[:prefix_len]

    full_tokens = prefix_tokens + needle_section + middle_filler_tokens + query_tokens + answer_tokens

    # Trim or pad to total_length
    if len(full_tokens) > total_length:
        # Trim prefix
        excess = len(full_tokens) - total_length
        prefix_tokens = prefix_tokens[excess:]
        full_tokens = prefix_tokens + needle_section + middle_filler_tokens + query_tokens + answer_tokens

    answer_start = len(full_tokens) - len(answer_tokens)
    answer_end = len(full_tokens)

    # Actual needle distance (in tokens from needle end to query start)
    needle_end_pos = len(prefix_tokens) + len(needle_section)
    query_start_pos = needle_end_pos + len(middle_filler_tokens)
    actual_distance = query_start_pos - needle_end_pos

    trial = NeedleTrial(
        name=template["name"],
        needle_text=needle_text if needle_present else "[absent]",
        query_text=query_text,
        answer_text=answer_text,
        needle_distance=actual_distance,
        needle_present=needle_present,
    )

    return full_tokens, answer_start, answer_end, trial


# ---------------------------------------------------------------------------
# Exit layer computation (reused logic from exit_layer_vs_longrange.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_exit_layers_for_positions(
    model, input_ids: torch.Tensor, positions: List[int]
) -> tuple:
    """
    Compute exit layer for specific token positions.

    positions: list of indices t where we want exit_layer for predicting token at t
               (i.e., using hidden state at position t-1 to predict token t).

    Returns:
        exit_layers: list of int, exit layer for each position
        final_correct: list of bool, whether final layer predicts correctly
        final_probs: list of float, probability of correct token at final layer
    """
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    num_layers = len(hidden_states) - 1

    norm = model.model.norm
    lm_head = model.lm_head

    exit_layers = []
    final_correct_list = []
    final_probs_list = []

    for pos in positions:
        if pos < 1 or pos >= input_ids.shape[1]:
            exit_layers.append(num_layers - 1)
            final_correct_list.append(False)
            final_probs_list.append(0.0)
            continue

        target_token = input_ids[0, pos].item()
        # Check each layer: use hidden state at pos-1 to predict token at pos
        preds = []
        for l in range(num_layers):
            h = hidden_states[l + 1][:, pos - 1 : pos, :]  # (1, 1, d)
            h_normed = norm(h)
            logits = lm_head(h_normed)[0, 0]  # (vocab,)
            preds.append(logits.argmax().item())

        # Find settling layer: earliest l where pred[l:] are all correct
        correct = [p == target_token for p in preds]
        el = num_layers - 1
        for l in range(num_layers - 1, -1, -1):
            if all(correct[l:]):
                el = l
        exit_layers.append(el)

        # Final layer stats
        final_logits = lm_head(norm(hidden_states[-1][:, pos - 1 : pos, :]))[0, 0]
        final_probs = F.softmax(final_logits.float(), dim=-1)
        final_correct_list.append(preds[-1] == target_token)
        final_probs_list.append(final_probs[target_token].item())

    return exit_layers, final_correct_list, final_probs_list


# ---------------------------------------------------------------------------
# Attention distance at specific positions
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_attention_distance_for_positions(
    model, input_ids: torch.Tensor, positions: List[int], sink_positions: int = 4
) -> List[float]:
    """
    Compute mean attention distance at specific query positions,
    averaged over all layers and heads, with sink filtering.
    """
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=False,
        output_attentions=True,
        use_cache=False,
    )
    attentions = outputs.attentions
    seq_len = input_ids.shape[1]
    device = input_ids.device

    pos_range = torch.arange(seq_len, device=device, dtype=torch.float32)

    # Sink mask
    sink_mask = torch.ones(seq_len, device=device)
    sink_mask[:sink_positions] = 0.0

    attn_dists = []
    for query_pos in positions:
        if query_pos < 1 or query_pos >= seq_len:
            attn_dists.append(0.0)
            continue

        # Use pos-1 (the input position that predicts token at pos)
        p = query_pos - 1
        total_dist = 0.0
        count = 0

        for attn_layer in attentions:
            attn = attn_layer[0].float()  # (num_heads, seq_len, seq_len)
            # Get attention from position p
            a = attn[:, p, :]  # (num_heads, seq_len)
            # Apply sink mask
            a = a * sink_mask.unsqueeze(0)
            a_sum = a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            a = a / a_sum

            # Weighted distance
            distances = (pos_range - p).abs()
            weighted = (a * distances.unsqueeze(0)).sum(dim=-1)  # (num_heads,)
            total_dist += weighted.sum().item()
            count += attn.shape[0]

        attn_dists.append(total_dist / max(count, 1))

    return attn_dists


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"  num_layers={num_layers}")

    # Experiment parameters
    distances = args.distances  # token distances to test
    n_trials = args.n_trials    # trials per template per distance per condition

    results_list: List[TrialResult] = []
    total_trials = len(NEEDLE_TEMPLATES) * len(distances) * 2 * n_trials  # 2 = present/absent
    trial_num = 0

    for template in NEEDLE_TEMPLATES:
        for distance in distances:
            for needle_present in [True, False]:
                for trial_i in range(n_trials):
                    trial_num += 1
                    condition = "present" if needle_present else "absent"
                    print(f"  [{trial_num}/{total_trials}] {template['name']} "
                          f"dist={distance} {condition} trial={trial_i+1}", end="\r")

                    tokens, ans_start, ans_end, trial_info = build_sequence(
                        tokenizer, template, distance, needle_present,
                        total_length=args.seq_length,
                    )

                    input_ids = torch.tensor([tokens], dtype=torch.long).to(model.device)
                    answer_positions = list(range(ans_start, ans_end))

                    # Exit layers for answer tokens
                    exit_layers, final_correct, final_probs = \
                        compute_exit_layers_for_positions(model, input_ids, answer_positions)

                    # Attention distance for answer tokens
                    attn_dists = compute_attention_distance_for_positions(
                        model, input_ids, answer_positions
                    )

                    result = TrialResult(
                        name=template["name"],
                        needle_distance=trial_info.needle_distance,
                        needle_present=needle_present,
                        answer_exit_layer=float(np.mean(exit_layers)),
                        answer_exit_layers=exit_layers,
                        answer_correct=all(final_correct),
                        full_context_prob=float(np.mean(final_probs)),
                        num_answer_tokens=len(answer_positions),
                    )
                    results_list.append(result)

    print()
    return results_list, num_layers


def analyze_results(results: List[TrialResult], num_layers: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

    # --- Group by condition ---
    present = [r for r in results if r.needle_present]
    absent = [r for r in results if not r.needle_present]

    el_present = np.array([r.answer_exit_layer for r in present])
    el_absent = np.array([r.answer_exit_layer for r in absent])

    print(f"\n=== Needle-in-Haystack Results ===")
    print(f"Total trials: {len(results)} ({len(present)} present, {len(absent)} absent)")
    print(f"\nMean exit layer (needle present):  {el_present.mean():.2f} +/- {el_present.std():.2f}")
    print(f"Mean exit layer (needle absent):   {el_absent.mean():.2f} +/- {el_absent.std():.2f}")

    # Test: do tokens need MORE layers when needle is absent?
    # (If the model can't find the fact, it should be "harder" = higher exit layer)
    if len(el_present) > 0 and len(el_absent) > 0:
        u_stat, p_val = stats.mannwhitneyu(el_present, el_absent, alternative="two-sided")
        print(f"Mann-Whitney U (present vs absent): U={u_stat:.0f}, p={p_val:.4f}")
    else:
        u_stat, p_val = np.nan, np.nan

    # --- Accuracy comparison ---
    acc_present = np.mean([r.answer_correct for r in present])
    acc_absent = np.mean([r.answer_correct for r in absent])
    print(f"\nAccuracy (needle present): {acc_present:.2%}")
    print(f"Accuracy (needle absent):  {acc_absent:.2%}")

    # --- Distance effect (within needle-present trials) ---
    distances_present = sorted(set(r.needle_distance for r in present))
    print(f"\nExit layer by needle distance (needle present):")
    distance_stats = {}
    for d in distances_present:
        trials_at_d = [r for r in present if r.needle_distance == d]
        els = [r.answer_exit_layer for r in trials_at_d]
        acc = np.mean([r.answer_correct for r in trials_at_d])
        mean_el = np.mean(els)
        print(f"  distance={d:>5d}: mean_exit_layer={mean_el:.2f}, accuracy={acc:.2%}, n={len(els)}")
        distance_stats[d] = {
            "mean_exit_layer": float(mean_el),
            "std_exit_layer": float(np.std(els)),
            "accuracy": float(acc),
            "n": len(els),
        }

    # Correlation: distance vs exit layer (within present trials)
    if len(distances_present) > 1:
        dist_arr = np.array([r.needle_distance for r in present])
        el_arr = np.array([r.answer_exit_layer for r in present])
        rho_dist, p_dist = stats.spearmanr(dist_arr, el_arr)
        print(f"\nSpearman(needle_distance, exit_layer) among present trials: "
              f"rho={rho_dist:.4f}, p={p_dist:.4f}")
    else:
        rho_dist, p_dist = np.nan, np.nan

    # --- Per-template breakdown ---
    template_results = {}
    for name in set(r.name for r in results):
        t_present = [r for r in present if r.name == name]
        t_absent = [r for r in absent if r.name == name]
        tp_el = np.mean([r.answer_exit_layer for r in t_present]) if t_present else np.nan
        ta_el = np.mean([r.answer_exit_layer for r in t_absent]) if t_absent else np.nan
        tp_acc = np.mean([r.answer_correct for r in t_present]) if t_present else np.nan
        ta_acc = np.mean([r.answer_correct for r in t_absent]) if t_absent else np.nan
        template_results[name] = {
            "present_mean_exit_layer": float(tp_el),
            "absent_mean_exit_layer": float(ta_el),
            "present_accuracy": float(tp_acc),
            "absent_accuracy": float(ta_acc),
            "n_present": len(t_present),
            "n_absent": len(t_absent),
        }
        print(f"\n  {name}: present_el={tp_el:.2f}, absent_el={ta_el:.2f}, "
              f"present_acc={tp_acc:.2%}, absent_acc={ta_acc:.2%}")

    # --- Plots ---
    # 1. Exit layer: present vs absent box plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([el_present, el_absent], labels=["Needle Present", "Needle Absent"], showfliers=True)
    ax.set_ylabel("Exit Layer (answer tokens)")
    ax.set_title("Exit Layer: Needle Present vs Absent")
    fig.savefig(os.path.join(output_dir, "plots", "nih_present_vs_absent.png"), dpi=120)
    plt.close(fig)

    # 2. Exit layer vs needle distance (present only)
    if len(distances_present) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        dist_groups = []
        dist_labels = []
        for d in distances_present:
            els = [r.answer_exit_layer for r in present if r.needle_distance == d]
            dist_groups.append(els)
            dist_labels.append(str(d))
        ax.boxplot(dist_groups, labels=dist_labels, showfliers=True)
        ax.set_xlabel("Needle Distance (tokens)")
        ax.set_ylabel("Exit Layer (answer tokens)")
        ax.set_title("Exit Layer vs Needle Distance")
        fig.savefig(os.path.join(output_dir, "plots", "nih_distance_vs_exit_layer.png"), dpi=120)
        plt.close(fig)

    # 3. Accuracy vs distance
    if len(distances_present) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        accs = [distance_stats[d]["accuracy"] for d in distances_present]
        ax.plot(distances_present, accs, "o-")
        ax.set_xlabel("Needle Distance (tokens)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Answer Accuracy vs Needle Distance")
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(os.path.join(output_dir, "plots", "nih_accuracy_vs_distance.png"), dpi=120)
        plt.close(fig)

    # --- Save JSON ---
    summary = {
        "num_trials": len(results),
        "num_layers": num_layers,
        "overall": {
            "present_mean_exit_layer": float(el_present.mean()),
            "absent_mean_exit_layer": float(el_absent.mean()),
            "present_std_exit_layer": float(el_present.std()),
            "absent_std_exit_layer": float(el_absent.std()),
            "mannwhitney_u": float(u_stat),
            "mannwhitney_p": float(p_val),
            "present_accuracy": float(acc_present),
            "absent_accuracy": float(acc_absent),
        },
        "distance_effect": {
            "spearman_rho": float(rho_dist),
            "spearman_p": float(p_dist),
            "by_distance": {str(d): v for d, v in distance_stats.items()},
        },
        "per_template": template_results,
        "interpretation": (
            "If needle-absent exit layers are significantly higher than present, "
            "it confirms that missing long-range information makes tokens harder. "
            "If exit layer increases with needle distance, it confirms that "
            "more distant information requires more layers to integrate. "
            "Both support the MemorySkip hypothesis."
        ),
    }

    out_path = os.path.join(output_dir, "needle_in_haystack_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Plots saved to {output_dir}/plots/")

    # Verdict
    if p_val < 0.05 and el_absent.mean() > el_present.mean():
        print("\nVERDICT: Needle absence increases exit layer (p<0.05). "
              "Long-range dependency causally affects token difficulty.")
    elif p_val < 0.05:
        print(f"\nVERDICT: Significant difference but unexpected direction. Investigate.")
    else:
        print(f"\nVERDICT: No significant difference (p={p_val:.4f}). "
              "Needle presence does not clearly affect exit layer.")


def main():
    parser = argparse.ArgumentParser(description="Needle-in-haystack controlled study")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Total sequence length for each trial")
    parser.add_argument("--distances", type=int, nargs="+", default=[64, 256, 512, 1024],
                        help="Needle distances to test (in tokens)")
    parser.add_argument("--n_trials", type=int, default=3,
                        help="Number of trials per template/distance/condition")
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    results, num_layers = run_experiment(args)
    analyze_results(results, num_layers, args.output_dir)


if __name__ == "__main__":
    main()
