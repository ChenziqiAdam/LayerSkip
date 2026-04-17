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
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
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

    # Compute predictions at each layer: preds[l, t] = predicted token id
    preds = torch.zeros(num_layers, seq_len, dtype=torch.long, device=input_ids.device)
    for l in range(num_layers):
        h = hidden_states[l + 1]  # (1, seq_len, d), skip embedding layer
        h_normed = norm(h)
        logits = lm_head(h_normed)  # (1, seq_len, vocab)
        preds[l] = logits[0].argmax(dim=-1)

    # correct[l, t] = 1 if preds[l, t] == targets[t]
    correct = (preds == targets.unsqueeze(0))  # (num_layers, seq_len)

    # exit_layer[t] = earliest l such that correct[l:, t].all()
    # Compute suffix-AND: suffix_correct[l, t] = correct[l:, t].all()
    # Walk from last layer to first
    suffix_correct = correct.clone()
    for l in range(num_layers - 2, -1, -1):
        suffix_correct[l] = correct[l] & suffix_correct[l + 1]

    # exit_layer[t] = first l where suffix_correct[l, t] is True
    # If no such l exists, assign num_layers - 1
    exit_layer = torch.full((seq_len,), num_layers - 1, dtype=torch.long, device=input_ids.device)
    # Process from last to first so that the earliest True wins
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
        attn_dist: (seq_len,) average attention distance (metric A)
        lr_mass:   (seq_len,) fraction of attention mass on positions >128 away (metric B)
    Both averaged over all layers and all heads.
    """
    outputs = model(
        input_ids=input_ids,
        output_hidden_states=False,
        output_attentions=True,
        use_cache=False,
    )
    attentions = outputs.attentions  # tuple of (num_layers,) each (1, num_heads, seq_len, seq_len)
    seq_len = input_ids.shape[1]
    device = input_ids.device

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    dist_matrix = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (seq_len, seq_len)

    sum_dist = torch.zeros(seq_len, device=device)
    sum_lr_mass = torch.zeros(seq_len, device=device)
    total_contributions = 0

    for attn in attentions:
        # attn: (1, num_heads, seq_len, seq_len)
        attn = attn[0].float()  # (num_heads, seq_len, seq_len)
        num_heads = attn.shape[0]

        # metric A: weighted average distance per query token, per head
        # attn[h, t, :] is the attention distribution over keys for query t
        # avg_dist[h, t] = sum_j( attn[h, t, j] * dist_matrix[t, j] )
        # dist_matrix is (seq_len, seq_len); broadcast over heads
        weighted_dist = (attn * dist_matrix.unsqueeze(0)).sum(dim=-1)  # (num_heads, seq_len)
        sum_dist += weighted_dist.sum(dim=0)  # sum over heads

        # metric B: fraction of attention on positions |t - j| > 128
        lr_mask = (dist_matrix > 128).float()  # (seq_len, seq_len)
        lr_weight = (attn * lr_mask.unsqueeze(0)).sum(dim=-1)  # (num_heads, seq_len)
        sum_lr_mass += lr_weight.sum(dim=0)

        total_contributions += num_heads

    attn_dist = (sum_dist / total_contributions).cpu()
    lr_mass = (sum_lr_mass / total_contributions).cpu()
    return attn_dist, lr_mass


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


# ---------------------------------------------------------------------------
# Step 6: Subgroup analysis
# ---------------------------------------------------------------------------

def subgroup_analysis(exit_layers, lr_mass, token_ids, tokenizer, freq_table):
    results = {}

    tok_array = np.array(token_ids)
    decoded = [tokenizer.decode([t]) for t in tok_array[:min(len(tok_array), 50000)]]

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

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...", end="\r")
        input_ids = torch.tensor([chunk], dtype=torch.long).to(model.device)
        # Targets are the next tokens: shift left by 1, use chunk[1:] as targets
        # We evaluate predictions at positions 0..seq_len-2 against targets chunk[1..seq_len-1]
        targets = torch.tensor(chunk[1:], dtype=torch.long).to(model.device)
        eval_input = input_ids  # full chunk as context

        # Exit layers (only for positions 0..seq_len-2)
        exit_layers_chunk = compute_exit_layers(model, eval_input, targets)
        # exit_layers_chunk has seq_len entries; position t predicts token t+1
        # Trim to seq_len-1 to align with targets
        exit_layers_chunk = exit_layers_chunk[:-1]  # (seq_len-1,)

        # Attention metrics
        attn_dist_chunk, lr_mass_chunk = compute_attention_metrics(model, eval_input)
        attn_dist_chunk = attn_dist_chunk[:-1]
        lr_mass_chunk = lr_mass_chunk[:-1]

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

    print()

    exit_layers = np.concatenate(all_exit_layers)
    attn_dist = np.concatenate(all_attn_dist)
    lr_mass = np.concatenate(all_lr_mass)
    kl_divs = np.concatenate(all_kl_divs)
    token_ids = np.array(all_token_ids)

    print(f"Total tokens analyzed: {len(exit_layers)}")
    print(f"Exit layer distribution: mean={exit_layers.mean():.1f}, "
          f"median={np.median(exit_layers):.0f}, max={exit_layers.max()}")

    # --- Correlation analysis ---
    rho_a, p_a = stats.spearmanr(exit_layers, attn_dist)
    rho_b, p_b = stats.spearmanr(exit_layers, lr_mass)
    rho_c, p_c = stats.spearmanr(exit_layers, kl_divs) if kl_divs.sum() > 0 else (float("nan"), float("nan"))

    print(f"\nSpearman correlations (exit_layer vs metric):")
    print(f"  Metric A (attn_dist):  rho={rho_a:.4f}, p={p_a:.2e}")
    print(f"  Metric B (lr_mass):    rho={rho_b:.4f}, p={p_b:.2e}")
    print(f"  Metric C (KL div):     rho={rho_c:.4f}, p={p_c:.2e}")

    # Partial Spearman controlling for log(token_freq)
    log_freqs = np.log(np.array([freq_table.get(int(t), 1e-10) for t in token_ids]) + 1e-10)
    prho_a, pp_a = partial_spearman(exit_layers.astype(float), attn_dist, log_freqs)
    prho_b, pp_b = partial_spearman(exit_layers.astype(float), lr_mass, log_freqs)

    print(f"\nPartial Spearman (controlling for log token frequency):")
    print(f"  Metric A (attn_dist):  rho={prho_a:.4f}, p={pp_a:.2e}")
    print(f"  Metric B (lr_mass):    rho={prho_b:.4f}, p={pp_b:.2e}")

    # Bucket analysis
    buckets_a = bucket_analysis(exit_layers, attn_dist, num_layers)
    buckets_b = bucket_analysis(exit_layers, lr_mass, num_layers)

    # Subgroup analysis
    subgroups = subgroup_analysis(exit_layers, lr_mass, token_ids, tokenizer, freq_table)

    # Plots
    make_plots(exit_layers, attn_dist, lr_mass, kl_divs, num_layers,
               os.path.join(args.output_dir, "plots"))

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
        "spearman": {
            "metric_a_attn_dist": {"rho": float(rho_a), "p": float(p_a)},
            "metric_b_lr_mass": {"rho": float(rho_b), "p": float(p_b)},
            "metric_c_kl_div": {"rho": float(rho_c), "p": float(p_c)},
        },
        "partial_spearman_controlling_log_freq": {
            "metric_a_attn_dist": {"rho": float(prho_a), "p": float(pp_a)},
            "metric_b_lr_mass": {"rho": float(prho_b), "p": float(pp_b)},
        },
        "bucket_analysis": {
            "metric_a": buckets_a,
            "metric_b": buckets_b,
        },
        "subgroup_analysis": subgroups,
        "interpretation": (
            "Success: rho > 0.3 for at least one metric after frequency control. "
            "Marginal: rho in [0.15, 0.3]. Pivot if rho < 0.15."
        ),
    }

    out_path = os.path.join(args.output_dir, "exit_layer_longrange_correlation.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print(f"Plots saved to {args.output_dir}/plots/")

    # Verdict
    best_rho = max(abs(prho_a), abs(prho_b))
    if best_rho > 0.3:
        print(f"\nVERDICT: SUCCESS (best partial rho={best_rho:.3f} > 0.3). MemorySkip is motivated.")
    elif best_rho > 0.15:
        print(f"\nVERDICT: MARGINAL (best partial rho={best_rho:.3f} in [0.15, 0.3]). Check subgroups.")
    else:
        print(f"\nVERDICT: WEAK (best partial rho={best_rho:.3f} < 0.15). Consider pivoting.")


if __name__ == "__main__":
    main()
