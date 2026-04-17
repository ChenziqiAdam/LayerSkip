"""
Experiment: Does exit layer correlate with long-range dependency?

Hypothesis: tokens needing more layers to predict correctly also attend more to
distant context. Tests this on WikiText-103 validation with Qwen3-1.7B.

Usage:
    python experiments/exit_layer_vs_longrange.py --model Qwen/Qwen3-1.7B
    python experiments/exit_layer_vs_longrange.py --model Qwen/Qwen3-1.7B --max_chunks 5  # smoke test
"""

import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()
    return model, tokenizer


def load_wikitext103_chunks(tokenizer, chunk_size: int = 2048, max_chunks: int = None):
    """Tokenize WikiText-103 validation into non-overlapping chunks of `chunk_size`."""
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="validation")
    full_text = "\n\n".join(t for t in dataset["text"] if t.strip())
    token_ids = tokenizer.encode(full_text, add_special_tokens=False)
    chunks = []
    for start in range(0, len(token_ids) - chunk_size, chunk_size):
        chunks.append(token_ids[start : start + chunk_size])
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks


# ---------------------------------------------------------------------------
# Step 2: Exit layer computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_exit_layers(model, input_ids: torch.Tensor, targets: torch.Tensor):
    """
    Returns exit_layer: LongTensor of shape (seq_len,).
    exit_layer[t] = earliest layer l where argmax(lm_head(norm(h_l))[t]) == targets[t]
    and that prediction holds for all subsequent layers.
    Tokens never predicted correctly get exit_layer = num_layers - 1.
    """
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states  # tuple of (num_layers+1,) each (1, seq_len, d)
    # hidden_states[0] = embedding output, hidden_states[1..L] = transformer layer outputs
    num_layers = len(hidden_states) - 1  # exclude embedding layer
    seq_len = input_ids.shape[1]

    norm = model.model.norm
    lm_head = model.lm_head

    # We predict token at position t+1 from hidden state at position t.
    # targets has seq_len-1 entries (chunk[1:]); evaluate positions 0..seq_len-2.
    eval_len = seq_len - 1  # number of (input, target) pairs

    # Compute predictions at each layer for positions 0..eval_len-1
    preds = torch.zeros(num_layers, eval_len, dtype=torch.long, device=input_ids.device)
    for l in range(num_layers):
        h = hidden_states[l + 1][:, :eval_len, :]  # (1, eval_len, d)
        h_normed = norm(h)
        logits = lm_head(h_normed)  # (1, eval_len, vocab)
        preds[l] = logits[0].argmax(dim=-1)

    # correct[l, t] = 1 if preds[l, t] == targets[t]
    correct = (preds == targets.unsqueeze(0))  # (num_layers, eval_len)

    # exit_layer[t] = earliest l such that correct[l:, t].all()
    suffix_correct = correct.clone()
    for l in range(num_layers - 2, -1, -1):
        suffix_correct[l] = correct[l] & suffix_correct[l + 1]

    # If no such l exists, assign num_layers - 1
    exit_layer = torch.full((eval_len,), num_layers - 1, dtype=torch.long, device=input_ids.device)
    for l in range(num_layers - 1, -1, -1):
        exit_layer[suffix_correct[l]] = l

    return exit_layer.cpu()


# ---------------------------------------------------------------------------
# Step 3: Long-range dependency metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_attention_metrics(model, input_ids: torch.Tensor):
    """
    Returns:
        attn_dist: (seq_len,) average attention distance (metric A), averaged over all layers/heads
        lr_mass:   (seq_len,) fraction of attention mass on positions >128 away (metric B), averaged
        per_layer_lr_mass: (num_layers, seq_len) lr_mass per layer (averaged over heads within layer)
    """
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=False,
        output_attentions=True,
        use_cache=False,
    )
    attentions = outputs.attentions  # tuple of (num_layers,) each (1, num_heads, seq_len, seq_len)
    num_layers = len(attentions)
    seq_len = input_ids.shape[1]
    device = input_ids.device

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist_matrix = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (seq_len, seq_len)
    lr_mask = (dist_matrix > 128).float()  # (seq_len, seq_len)

    sum_dist = torch.zeros(seq_len, device=device)
    sum_lr_mass = torch.zeros(seq_len, device=device)
    per_layer_lr_mass = torch.zeros(num_layers, seq_len, device=device)
    total_contributions = 0

    for layer_idx, attn in enumerate(attentions):
        # attn: (1, num_heads, seq_len, seq_len)
        attn = attn[0].float()  # (num_heads, seq_len, seq_len)
        num_heads = attn.shape[0]

        # metric A: weighted average distance per query token, per head
        weighted_dist = (attn * dist_matrix.unsqueeze(0)).sum(dim=-1)  # (num_heads, seq_len)
        sum_dist += weighted_dist.sum(dim=0)

        # metric B: fraction of attention on positions |t - j| > 128
        lr_weight = (attn * lr_mask.unsqueeze(0)).sum(dim=-1)  # (num_heads, seq_len)
        sum_lr_mass += lr_weight.sum(dim=0)

        # Per-layer lr_mass: average over heads for this layer
        per_layer_lr_mass[layer_idx] = lr_weight.mean(dim=0)

        total_contributions += num_heads

    attn_dist = (sum_dist / total_contributions).cpu()
    lr_mass = (sum_lr_mass / total_contributions).cpu()
    per_layer_lr_mass = per_layer_lr_mass.cpu()
    return attn_dist, lr_mass, per_layer_lr_mass


def compute_layer_specific_metrics(exit_layers_chunk, per_layer_lr_mass):
    """
    Given per-token exit layers and per-layer lr_mass, compute:
        lr_mass_at_exit:  lr_mass at each token's own exit layer
        lr_mass_at_final: lr_mass at the final layer for each token
        lr_mass_growth:   lr_mass(final) - lr_mass(early), where early = layer num_layers//4
    All returned as (eval_len,) numpy arrays.
    """
    # per_layer_lr_mass: (num_layers, seq_len) — trim to eval_len (seq_len-1)
    num_layers, seq_len = per_layer_lr_mass.shape
    eval_len = seq_len - 1
    plm = per_layer_lr_mass[:, :eval_len].numpy()  # (num_layers, eval_len)
    el = exit_layers_chunk.numpy()  # (eval_len,)

    token_indices = np.arange(eval_len)

    # lr_mass at each token's exit layer
    lr_mass_at_exit = plm[el, token_indices]

    # lr_mass at the final layer
    lr_mass_at_final = plm[-1, :]

    # lr_mass growth: final layer minus an early layer (layer num_layers//4)
    early_layer = num_layers // 4
    lr_mass_growth = plm[-1, :] - plm[early_layer, :]

    return lr_mass_at_exit, lr_mass_at_final, lr_mass_growth


@torch.no_grad()
def compute_kl_metric(model, input_ids: torch.Tensor, trunc_len: int = 256):
    """
    Metric C: KL(P_full || P_trunc) for each token.
    Returns kl_div: (seq_len,) tensor.
    Tokens within the first trunc_len positions get KL = 0 (truncation has no effect there).
    """
    seq_len = input_ids.shape[1]

    # Full context logits
    full_logits = model(input_ids=input_ids, use_cache=False).logits[0]  # (seq_len, vocab)
    full_probs = F.softmax(full_logits.float(), dim=-1)

    kl_divs = torch.zeros(seq_len)

    # For each position t > trunc_len, run truncated forward pass
    # Batch this: use fixed truncation from position max(0, t - trunc_len) to t
    # For efficiency, we use a sliding window approach:
    # group tokens by their truncation window start
    # Simple approach: for tokens at position >= trunc_len, run one pass with just the last trunc_len tokens
    if seq_len > trunc_len:
        trunc_input = input_ids[:, -trunc_len:]
        trunc_logits = model(trunc_input, use_cache=False).logits[0]  # (trunc_len, vocab)
        trunc_probs = F.softmax(trunc_logits.float(), dim=-1)

        # Align: truncated pass covers the last trunc_len positions of the full sequence
        offset = seq_len - trunc_len
        for i in range(trunc_len):
            t = offset + i
            kl = F.kl_div(
                trunc_probs[i].log(),
                full_probs[t],
                reduction="sum",
            ).item()
            kl_divs[t] = max(kl, 0.0)  # numerical safety

    return kl_divs


# ---------------------------------------------------------------------------
# Step 4: Token-level frequency estimation
# ---------------------------------------------------------------------------

def build_frequency_table(chunks):
    """Count token frequencies across all chunks."""
    counter = Counter()
    for chunk in chunks:
        counter.update(chunk)
    total = sum(counter.values())
    freq = {tok: count / total for tok, count in counter.items()}
    return freq


# ---------------------------------------------------------------------------
# Step 5: Correlation analysis
# ---------------------------------------------------------------------------

def partial_spearman(x, y, covariate):
    """Spearman correlation between x and y after linearly removing covariate."""
    def residuals(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)

    x_res = residuals(stats.rankdata(x), covariate)
    y_res = residuals(stats.rankdata(y), covariate)
    rho, pval = stats.spearmanr(x_res, y_res)
    return rho, pval


def bucket_analysis(exit_layers, metric, num_layers):
    """Split tokens into early/mid/late exit buckets and run Mann-Whitney U."""
    third = num_layers // 3
    early = metric[exit_layers <= third]
    mid = metric[(exit_layers > third) & (exit_layers <= 2 * third)]
    late = metric[exit_layers > 2 * third]
    stat_em, p_em = stats.mannwhitneyu(early, mid, alternative="two-sided") if len(early) and len(mid) else (np.nan, np.nan)
    stat_ml, p_ml = stats.mannwhitneyu(mid, late, alternative="two-sided") if len(mid) and len(late) else (np.nan, np.nan)
    return {
        "early_mean": float(np.mean(early)) if len(early) else None,
        "mid_mean": float(np.mean(mid)) if len(mid) else None,
        "late_mean": float(np.mean(late)) if len(late) else None,
        "mannwhitney_early_mid_p": float(p_em),
        "mannwhitney_mid_late_p": float(p_ml),
    }


def make_plots(exit_layers, attn_dist, lr_mass, kl_divs, num_layers, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Scatter: exit layer vs attention distance
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(exit_layers, attn_dist, alpha=0.05, s=2, rasterized=True)
    ax.set_xlabel("Exit Layer")
    ax.set_ylabel("Mean Attention Distance")
    ax.set_title("Exit Layer vs Attention Distance")
    fig.savefig(os.path.join(output_dir, "exit_layer_vs_attn_dist.png"), dpi=120)
    plt.close(fig)

    # Scatter: exit layer vs long-range mass
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(exit_layers, lr_mass, alpha=0.05, s=2, rasterized=True)
    ax.set_xlabel("Exit Layer")
    ax.set_ylabel("Long-Range Attention Mass (>128 tokens)")
    ax.set_title("Exit Layer vs Long-Range Attention Mass")
    fig.savefig(os.path.join(output_dir, "exit_layer_vs_lr_mass.png"), dpi=120)
    plt.close(fig)

    # Box plot: distribution of lr_mass by exit bucket
    third = num_layers // 3
    groups = [
        lr_mass[exit_layers <= third],
        lr_mass[(exit_layers > third) & (exit_layers <= 2 * third)],
        lr_mass[exit_layers > 2 * third],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(groups, labels=["Early", "Mid", "Late"], showfliers=False)
    ax.set_xlabel("Exit Bucket")
    ax.set_ylabel("Long-Range Attention Mass")
    ax.set_title("Long-Range Mass by Exit Layer Bucket")
    fig.savefig(os.path.join(output_dir, "bucket_boxplot.png"), dpi=120)
    plt.close(fig)

    if kl_divs is not None and kl_divs.sum() > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(exit_layers, kl_divs, alpha=0.05, s=2, rasterized=True)
        ax.set_xlabel("Exit Layer")
        ax.set_ylabel("KL Divergence (full vs truncated)")
        ax.set_title("Exit Layer vs Counterfactual KL")
        fig.savefig(os.path.join(output_dir, "exit_layer_vs_kl.png"), dpi=120)
        plt.close(fig)


def make_layer_specific_plots(exit_layers, lr_mass_at_exit, lr_mass_at_final, lr_mass_growth,
                               num_layers, output_dir):
    """Plots for layer-specific long-range metrics."""
    os.makedirs(output_dir, exist_ok=True)
    third = num_layers // 3

    # Scatter: exit layer vs lr_mass at final layer
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(exit_layers, lr_mass_at_final, alpha=0.05, s=2, rasterized=True)
    ax.set_xlabel("Exit Layer")
    ax.set_ylabel("LR Attention Mass @ Final Layer")
    ax.set_title("Exit Layer vs LR Mass at Final Layer")
    fig.savefig(os.path.join(output_dir, "exit_layer_vs_lr_mass_final.png"), dpi=120)
    plt.close(fig)

    # Scatter: exit layer vs lr_mass growth
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(exit_layers, lr_mass_growth, alpha=0.05, s=2, rasterized=True)
    ax.set_xlabel("Exit Layer")
    ax.set_ylabel("LR Mass Growth (final - early layer)")
    ax.set_title("Exit Layer vs LR Mass Growth Across Depth")
    fig.savefig(os.path.join(output_dir, "exit_layer_vs_lr_mass_growth.png"), dpi=120)
    plt.close(fig)

    # Box plot: lr_mass at final layer by exit bucket
    groups_final = [
        lr_mass_at_final[exit_layers <= third],
        lr_mass_at_final[(exit_layers > third) & (exit_layers <= 2 * third)],
        lr_mass_at_final[exit_layers > 2 * third],
    ]
    groups_growth = [
        lr_mass_growth[exit_layers <= third],
        lr_mass_growth[(exit_layers > third) & (exit_layers <= 2 * third)],
        lr_mass_growth[exit_layers > 2 * third],
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.boxplot(groups_final, labels=["Early", "Mid", "Late"], showfliers=False)
    ax1.set_xlabel("Exit Bucket")
    ax1.set_ylabel("LR Mass @ Final Layer")
    ax1.set_title("Final-Layer LR Mass by Exit Bucket")
    ax2.boxplot(groups_growth, labels=["Early", "Mid", "Late"], showfliers=False)
    ax2.set_xlabel("Exit Bucket")
    ax2.set_ylabel("LR Mass Growth")
    ax2.set_title("LR Mass Growth by Exit Bucket")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "bucket_boxplot_layer_specific.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 6: Subgroup analysis
# ---------------------------------------------------------------------------

def subgroup_analysis(exit_layers, lr_mass, token_ids, tokenizer, freq_table):
    results = {}

    tok_array = np.array(token_ids)
    n = len(tok_array)
    decoded = [tokenizer.decode([t]) for t in tok_array]

    # Proper nouns / numbers: token starts with capital or digit
    is_rare_start = np.array([
        len(d) > 0 and (d[0].isupper() or d[0].isdigit() or (d[0] == " " and len(d) > 1 and (d[1].isupper() or d[1].isdigit())))
        for d in decoded
    ], dtype=bool)

    # Rare tokens: bottom 20% by frequency
    freqs = np.array([freq_table.get(t, 1e-10) for t in tok_array])
    freq_threshold = np.percentile(freqs, 20)
    is_rare = freqs <= freq_threshold

    for name, mask in [("capitalized_or_digit", is_rare_start), ("rare_tokens", is_rare)]:
        if mask.sum() < 10:
            continue
        rho, pval = stats.spearmanr(exit_layers[mask], lr_mass[mask])
        results[name] = {
            "n": int(mask.sum()),
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "mean_lr_mass": float(lr_mass[mask].mean()),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max_chunks", type=int, default=None, help="Limit chunks for testing")
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--lr_threshold", type=int, default=128, help="Distance threshold for long-range")
    parser.add_argument("--kl_sample_rate", type=float, default=0.1, help="Fraction of chunks to compute KL for")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--no_kl", action="store_true", help="Skip KL metric entirely")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "plots"), exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    num_layers = model.config.num_hidden_layers
    print(f"  num_layers={num_layers}")

    print("Loading WikiText-103 validation...")
    chunks = load_wikitext103_chunks(tokenizer, args.chunk_size, args.max_chunks)
    print(f"  {len(chunks)} chunks of {args.chunk_size} tokens each")

    # Build frequency table
    freq_table = build_frequency_table(chunks)

    all_exit_layers = []
    all_attn_dist = []
    all_lr_mass = []
    all_kl_divs = []
    all_token_ids = []
    all_lr_mass_at_exit = []
    all_lr_mass_at_final = []
    all_lr_mass_growth = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...", end="\r")
        input_ids = torch.tensor([chunk], dtype=torch.long).to(model.device)
        # Targets are the next tokens: shift left by 1, use chunk[1:] as targets
        # We evaluate predictions at positions 0..seq_len-2 against targets chunk[1..seq_len-1]
        targets = torch.tensor(chunk[1:], dtype=torch.long).to(model.device)
        eval_input = input_ids  # full chunk as context

        # Exit layers: returns (seq_len-1,) array, already aligned with targets
        exit_layers_chunk = compute_exit_layers(model, eval_input, targets)

        # Attention metrics: (seq_len,) → trim last position to match eval_len
        attn_dist_chunk, lr_mass_chunk, per_layer_lr_mass = compute_attention_metrics(model, eval_input)
        attn_dist_chunk = attn_dist_chunk[:-1]
        lr_mass_chunk = lr_mass_chunk[:-1]

        # Layer-specific metrics
        lr_at_exit, lr_at_final, lr_growth = compute_layer_specific_metrics(
            exit_layers_chunk, per_layer_lr_mass
        )

        # KL metric (optional, sampled)
        if not args.no_kl and (i / len(chunks)) < args.kl_sample_rate:
            kl_chunk = compute_kl_metric(model, eval_input)
            kl_chunk = kl_chunk[:-1]
        else:
            kl_chunk = torch.zeros(len(chunk) - 1)

        all_exit_layers.append(exit_layers_chunk.numpy())
        all_attn_dist.append(attn_dist_chunk.numpy())
        all_lr_mass.append(lr_mass_chunk.numpy())
        all_kl_divs.append(kl_chunk.numpy())
        all_token_ids.extend(chunk[:-1])
        all_lr_mass_at_exit.append(lr_at_exit)
        all_lr_mass_at_final.append(lr_at_final)
        all_lr_mass_growth.append(lr_growth)

    print()

    exit_layers = np.concatenate(all_exit_layers)
    attn_dist = np.concatenate(all_attn_dist)
    lr_mass = np.concatenate(all_lr_mass)
    kl_divs = np.concatenate(all_kl_divs)
    token_ids = np.array(all_token_ids)
    lr_mass_at_exit = np.concatenate(all_lr_mass_at_exit)
    lr_mass_at_final = np.concatenate(all_lr_mass_at_final)
    lr_mass_growth = np.concatenate(all_lr_mass_growth)

    print(f"Total tokens analyzed: {len(exit_layers)}")
    print(f"Exit layer distribution: mean={exit_layers.mean():.1f}, "
          f"median={np.median(exit_layers):.0f}, max={exit_layers.max()}")

    # --- Correlation analysis: original averaged metrics ---
    rho_a, p_a = stats.spearmanr(exit_layers, attn_dist)
    rho_b, p_b = stats.spearmanr(exit_layers, lr_mass)
    rho_c, p_c = stats.spearmanr(exit_layers, kl_divs) if kl_divs.sum() > 0 else (float("nan"), float("nan"))

    print(f"\nSpearman correlations (exit_layer vs metric) — averaged over all layers:")
    print(f"  Metric A (attn_dist):  rho={rho_a:.4f}, p={p_a:.2e}")
    print(f"  Metric B (lr_mass):    rho={rho_b:.4f}, p={p_b:.2e}")
    print(f"  Metric C (KL div):     rho={rho_c:.4f}, p={p_c:.2e}")

    # --- Correlation analysis: layer-specific metrics ---
    rho_exit, p_exit = stats.spearmanr(exit_layers, lr_mass_at_exit)
    rho_final, p_final = stats.spearmanr(exit_layers, lr_mass_at_final)
    rho_growth, p_growth = stats.spearmanr(exit_layers, lr_mass_growth)

    print(f"\nSpearman correlations — layer-specific metrics:")
    print(f"  LR mass @ exit layer:  rho={rho_exit:.4f}, p={p_exit:.2e}")
    print(f"  LR mass @ final layer: rho={rho_final:.4f}, p={p_final:.2e}")
    print(f"  LR mass growth (final - early): rho={rho_growth:.4f}, p={p_growth:.2e}")

    # Partial Spearman controlling for log(token_freq)
    log_freqs = np.log(np.array([freq_table.get(int(t), 1e-10) for t in token_ids]) + 1e-10)
    prho_a, pp_a = partial_spearman(exit_layers.astype(float), attn_dist, log_freqs)
    prho_b, pp_b = partial_spearman(exit_layers.astype(float), lr_mass, log_freqs)
    prho_exit, pp_exit = partial_spearman(exit_layers.astype(float), lr_mass_at_exit, log_freqs)
    prho_final, pp_final = partial_spearman(exit_layers.astype(float), lr_mass_at_final, log_freqs)
    prho_growth, pp_growth = partial_spearman(exit_layers.astype(float), lr_mass_growth, log_freqs)

    print(f"\nPartial Spearman (controlling for log token frequency):")
    print(f"  Metric A (attn_dist):  rho={prho_a:.4f}, p={pp_a:.2e}")
    print(f"  Metric B (lr_mass):    rho={prho_b:.4f}, p={pp_b:.2e}")
    print(f"  LR mass @ exit layer:  rho={prho_exit:.4f}, p={pp_exit:.2e}")
    print(f"  LR mass @ final layer: rho={prho_final:.4f}, p={pp_final:.2e}")
    print(f"  LR mass growth:        rho={prho_growth:.4f}, p={pp_growth:.2e}")

    # Bucket analysis
    buckets_a = bucket_analysis(exit_layers, attn_dist, num_layers)
    buckets_b = bucket_analysis(exit_layers, lr_mass, num_layers)
    buckets_final = bucket_analysis(exit_layers, lr_mass_at_final, num_layers)
    buckets_growth = bucket_analysis(exit_layers, lr_mass_growth, num_layers)

    # Subgroup analysis
    subgroups = subgroup_analysis(exit_layers, lr_mass, token_ids, tokenizer, freq_table)

    # Plots
    make_plots(exit_layers, attn_dist, lr_mass, kl_divs, num_layers,
               os.path.join(args.output_dir, "plots"))
    make_layer_specific_plots(exit_layers, lr_mass_at_exit, lr_mass_at_final, lr_mass_growth,
                              num_layers, os.path.join(args.output_dir, "plots"))

    # Save results
    results = {
        "model": args.model,
        "num_chunks": len(chunks),
        "num_tokens": int(len(exit_layers)),
        "num_layers": num_layers,
        "exit_layer_stats": {
            "mean": float(exit_layers.mean()),
            "median": float(np.median(exit_layers)),
            "std": float(exit_layers.std()),
        },
        "spearman_averaged": {
            "metric_a_attn_dist": {"rho": float(rho_a), "p": float(p_a)},
            "metric_b_lr_mass": {"rho": float(rho_b), "p": float(p_b)},
            "metric_c_kl_div": {"rho": float(rho_c), "p": float(p_c)},
        },
        "spearman_layer_specific": {
            "lr_mass_at_exit": {"rho": float(rho_exit), "p": float(p_exit)},
            "lr_mass_at_final": {"rho": float(rho_final), "p": float(p_final)},
            "lr_mass_growth": {"rho": float(rho_growth), "p": float(p_growth)},
        },
        "partial_spearman_controlling_log_freq": {
            "metric_a_attn_dist": {"rho": float(prho_a), "p": float(pp_a)},
            "metric_b_lr_mass": {"rho": float(prho_b), "p": float(pp_b)},
            "lr_mass_at_exit": {"rho": float(prho_exit), "p": float(pp_exit)},
            "lr_mass_at_final": {"rho": float(prho_final), "p": float(pp_final)},
            "lr_mass_growth": {"rho": float(prho_growth), "p": float(pp_growth)},
        },
        "bucket_analysis": {
            "metric_a": buckets_a,
            "metric_b": buckets_b,
            "lr_mass_at_final": buckets_final,
            "lr_mass_growth": buckets_growth,
        },
        "subgroup_analysis": subgroups,
        "interpretation": (
            "Success: rho > 0.3 for at least one metric after frequency control. "
            "Marginal: rho in [0.15, 0.3]. Pivot if rho < 0.15. "
            "Layer-specific metrics (at_exit, at_final, growth) test the stronger claim "
            "that hard tokens specifically rely on long-range context at the layers that matter."
        ),
    }

    out_path = os.path.join(args.output_dir, "exit_layer_longrange_correlation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Plots saved to {args.output_dir}/plots/")

    # Verdict — consider both averaged and layer-specific metrics
    all_partial_rhos = [abs(prho_a), abs(prho_b), abs(prho_exit), abs(prho_final), abs(prho_growth)]
    best_rho = max(all_partial_rhos)
    best_names = ["attn_dist", "lr_mass_avg", "lr_mass@exit", "lr_mass@final", "lr_mass_growth"]
    best_name = best_names[np.argmax(all_partial_rhos)]
    if best_rho > 0.3:
        print(f"\nVERDICT: SUCCESS (best partial rho={best_rho:.3f} [{best_name}] > 0.3). MemorySkip is motivated.")
    elif best_rho > 0.15:
        print(f"\nVERDICT: MARGINAL (best partial rho={best_rho:.3f} [{best_name}] in [0.15, 0.3]). Check subgroups.")
    else:
        print(f"\nVERDICT: WEAK (best partial rho={best_rho:.3f} [{best_name}] < 0.15). Consider pivoting.")


if __name__ == "__main__":
    main()
