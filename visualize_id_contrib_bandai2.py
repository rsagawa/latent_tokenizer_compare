#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    Line2D = None
    HAS_MATPLOTLIB = False


DEFAULT_NAMES = ["MotionGPT", "m2dm", "proposed", "latent_tokenizer"]


def _to_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _task_dir(base_dir: Path, task: str, name: str, out_suffix: str) -> Path:
    return base_dir / f"{task}2_{name}" / out_suffix


def load_all_data(
    base_dir: Path,
    out_suffix: str,
    attr_suffix: str,
    names: List[str],
) -> Tuple[
    Dict[str, Dict[str, Dict[str, object]]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
    Dict[str, Dict[str, List[Dict[str, str]]]],
]:
    summaries: Dict[str, Dict[str, Dict[str, object]]] = {"actionrec": {}, "retrieval": {}}
    perturb_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = {"actionrec": {}, "retrieval": {}}
    class_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = {"actionrec": {}, "retrieval": {}}
    type_class_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = {"actionrec": {}, "retrieval": {}}
    attr_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = {"actionrec": {}, "retrieval": {}}
    attr_sample_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = {"actionrec": {}, "retrieval": {}}

    for task in ("actionrec", "retrieval"):
        for name in names:
            d = _task_dir(base_dir, task, name, out_suffix)
            summary_path = d / "id_contrib_summary.json"
            class_path = d / "id_class_stats.csv"
            type_class_path = d / "id_type_class_stats.csv"
            if not summary_path.exists() or not class_path.exists() or not type_class_path.exists():
                print(f"[warn] missing files: {d}")
                continue
            summaries[task][name] = _load_json(summary_path)
            perturb_path = d / "id_perturbation.csv"
            perturb_rows[task][name] = _load_csv_rows(perturb_path) if perturb_path.exists() else []
            class_rows[task][name] = _load_csv_rows(class_path)
            type_class_rows[task][name] = _load_csv_rows(type_class_path)
            attr_dir = base_dir / f"{task}2_{name}" / (attr_suffix if str(attr_suffix).strip() else out_suffix)
            attr_path = attr_dir / "id_attribution.csv"
            if attr_path.exists():
                attr_rows[task][name] = _load_csv_rows(attr_path)
            attr_sample_path = attr_dir / "id_attribution_per_sample.csv"
            if attr_sample_path.exists():
                attr_sample_rows[task][name] = _load_csv_rows(attr_sample_path)
    return summaries, perturb_rows, class_rows, type_class_rows, attr_rows, attr_sample_rows


def plot_base_metrics(
    summaries: Dict[str, Dict[str, Dict[str, object]]],
    names: List[str],
    out_path: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print("[warn] matplotlib not found: skip base metrics plot")
        return
    models = [n for n in names if n in summaries["actionrec"] and n in summaries["retrieval"]]
    if not models:
        print("[warn] no common models found for base metrics plot")
        return

    action_metric_keys = ["acc", "macro_f1", "mean_true_prob"]
    retrieval_metric_keys = ["t2m_R@1", "m2t_R@1", "mean_R@1"]
    action_vals = []
    retrieval_vals = []
    for m in models:
        action_vals.append(
            [
                _to_float(summaries["actionrec"][m]["base_metrics"].get("acc")),
                _to_float(summaries["actionrec"][m]["base_metrics"].get("macro_f1")),
                _to_float(summaries["actionrec"][m]["base_metrics"].get("mean_true_prob")),
            ]
        )
        t2m_r1 = _to_float(summaries["retrieval"][m]["base_metrics"]["t2m"].get("R@1"))
        m2t_r1 = _to_float(summaries["retrieval"][m]["base_metrics"]["m2t"].get("R@1"))
        retrieval_vals.append([t2m_r1, m2t_r1, (t2m_r1 + m2t_r1) / 2.0])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x = list(range(len(models)))
    width = 0.24

    for i, key in enumerate(action_metric_keys):
        ys = [row[i] for row in action_vals]
        xs = [xx + (i - 1) * width for xx in x]
        axes[0].bar(xs, ys, width=width, label=key)
    axes[0].set_title("ActionRec Base Metrics")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    for i, key in enumerate(retrieval_metric_keys):
        ys = [row[i] for row in retrieval_vals]
        xs = [xx + (i - 1) * width for xx in x]
        axes[1].bar(xs, ys, width=width, label=key)
    axes[1].set_title("Retrieval Base Metrics")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _split_class_name(class_name: str) -> Tuple[str, str]:
    if "__" in class_name:
        v, a = class_name.split("__", 1)
        return v, a
    return class_name, "unknown"


def _extract_verb_adj_from_row(r: Dict[str, str]) -> Tuple[str, str]:
    verb = str(r.get("verb", "")).strip()
    adj = str(r.get("adjective", "")).strip()
    if not verb or not adj:
        cls = str(r.get("class_name", ""))
        v2, a2 = _split_class_name(cls)
        verb = verb or v2
        adj = adj or a2
    return verb, adj


def _aggregate_attr_samples_by_mode(
    sample_rows: List[Dict[str, str]],
    mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
) -> Tuple[
    Dict[int, float],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, Dict[str, float]]],
]:
    total_by_id: Dict[int, float] = {}
    verb_by_id: Dict[int, Dict[str, float]] = {}
    adj_by_id: Dict[int, Dict[str, float]] = {}
    verb_adj_by_id: Dict[int, Dict[str, Dict[str, float]]] = {}
    m = str(mode).strip()
    use_top_p = 0.0 < float(sample_top_p) < 1.0

    def _build_scored(rows: List[Dict[str, str]]) -> List[Tuple[int, str, str, float]]:
        scored: List[Tuple[int, str, str, float]] = []
        for r in rows:
            iid = int(_to_float(r.get("item_id"), 0.0))
            w = _to_float(r.get("abs_attr_norm"), abs(_to_float(r.get("signed_attr_norm"), 0.0)))
            if w <= 0.0:
                continue
            verb, adj = _extract_verb_adj_from_row(r)
            scored.append((iid, verb, adj, w))
        return sorted(scored, key=lambda x: (-x[3], x[0]))

    def _apply_top_p(scored: List[Tuple[int, str, str, float]]) -> List[Tuple[int, str, str, float]]:
        if not use_top_p:
            return scored
        p = min(1.0, max(1e-8, float(sample_top_p)))
        cum = 0.0
        out: List[Tuple[int, str, str, float]] = []
        for t in scored:
            out.append(t)
            cum += float(t[3])
            if cum >= p:
                break
        return out

    if m == "sample_topk_count":
        by_sample: Dict[str, List[Dict[str, str]]] = {}
        for r in sample_rows:
            mid = str(r.get("mid", ""))
            by_sample.setdefault(mid, []).append(r)
        for _mid, rows in by_sample.items():
            scored = _build_scored(rows)
            scored = _apply_top_p(scored)
            if int(sample_top_k_ids) > 0:
                scored = scored[: int(sample_top_k_ids)]
            for iid, verb, adj, _w in scored:
                total_by_id[iid] = total_by_id.get(iid, 0.0) + 1.0
                verb_by_id.setdefault(iid, {})
                adj_by_id.setdefault(iid, {})
                verb_adj_by_id.setdefault(iid, {})
                verb_adj_by_id[iid].setdefault(verb, {})
                verb_by_id[iid][verb] = verb_by_id[iid].get(verb, 0.0) + 1.0
                adj_by_id[iid][adj] = adj_by_id[iid].get(adj, 0.0) + 1.0
                verb_adj_by_id[iid][verb][adj] = verb_adj_by_id[iid][verb].get(adj, 0.0) + 1.0
        return total_by_id, verb_by_id, adj_by_id, verb_adj_by_id

    if m == "sample_top_p_count":
        by_sample: Dict[str, List[Dict[str, str]]] = {}
        for r in sample_rows:
            mid = str(r.get("mid", ""))
            by_sample.setdefault(mid, []).append(r)
        for _mid, rows in by_sample.items():
            scored = _build_scored(rows)
            selected = _apply_top_p(scored)
            for iid, verb, adj, _w in selected:
                total_by_id[iid] = total_by_id.get(iid, 0.0) + 1.0
                verb_by_id.setdefault(iid, {})
                adj_by_id.setdefault(iid, {})
                verb_adj_by_id.setdefault(iid, {})
                verb_adj_by_id[iid].setdefault(verb, {})
                verb_by_id[iid][verb] = verb_by_id[iid].get(verb, 0.0) + 1.0
                adj_by_id[iid][adj] = adj_by_id[iid].get(adj, 0.0) + 1.0
                verb_adj_by_id[iid][verb][adj] = verb_adj_by_id[iid][verb].get(adj, 0.0) + 1.0
        return total_by_id, verb_by_id, adj_by_id, verb_adj_by_id

    # default: abs_attr_norm sum
    if use_top_p:
        by_sample: Dict[str, List[Dict[str, str]]] = {}
        for r in sample_rows:
            mid = str(r.get("mid", ""))
            by_sample.setdefault(mid, []).append(r)
        for _mid, rows in by_sample.items():
            scored = _build_scored(rows)
            selected = _apply_top_p(scored)
            for iid, verb, adj, w in selected:
                total_by_id[iid] = total_by_id.get(iid, 0.0) + w
                verb_by_id.setdefault(iid, {})
                adj_by_id.setdefault(iid, {})
                verb_adj_by_id.setdefault(iid, {})
                verb_adj_by_id[iid].setdefault(verb, {})
                verb_by_id[iid][verb] = verb_by_id[iid].get(verb, 0.0) + w
                adj_by_id[iid][adj] = adj_by_id[iid].get(adj, 0.0) + w
                verb_adj_by_id[iid][verb][adj] = verb_adj_by_id[iid][verb].get(adj, 0.0) + w
    else:
        for r in sample_rows:
            iid = int(_to_float(r.get("item_id"), 0.0))
            w = _to_float(r.get("abs_attr_norm"), abs(_to_float(r.get("signed_attr_norm"), 0.0)))
            if w <= 0.0:
                continue
            verb, adj = _extract_verb_adj_from_row(r)
            total_by_id[iid] = total_by_id.get(iid, 0.0) + w
            verb_by_id.setdefault(iid, {})
            adj_by_id.setdefault(iid, {})
            verb_adj_by_id.setdefault(iid, {})
            verb_adj_by_id[iid].setdefault(verb, {})
            verb_by_id[iid][verb] = verb_by_id[iid].get(verb, 0.0) + w
            adj_by_id[iid][adj] = adj_by_id[iid].get(adj, 0.0) + w
            verb_adj_by_id[iid][verb][adj] = verb_adj_by_id[iid][verb].get(adj, 0.0) + w
    return total_by_id, verb_by_id, adj_by_id, verb_adj_by_id


def _normalized_entropy(counts: Dict[str, float]) -> float:
    vals = [v for v in counts.values() if v > 0.0]
    if not vals:
        return 0.0
    s = sum(vals)
    if s <= 0.0:
        return 0.0
    ps = [v / s for v in vals]
    if len(ps) <= 1:
        return 0.0
    h = 0.0
    for p in ps:
        if p > 0.0:
            h -= p * math.log(p)
    hmax = math.log(len(ps))
    if hmax <= 0.0:
        return 0.0
    return h / hmax


def _bias_bucket(concentration: float, top_share: float) -> str:
    if concentration >= 0.75 and top_share >= 0.60:
        return "single"
    if concentration >= 0.45:
        return "few"
    return "broad"


def _compute_adj_bias_within_verbs(
    verb_adj_w: Dict[str, Dict[str, float]],
) -> Tuple[str, float, float, int]:
    verb_maps: List[Dict[str, float]] = []
    for _verb, amap in verb_adj_w.items():
        clean = {k: float(v) for k, v in amap.items() if float(v) > 0.0}
        if sum(clean.values()) > 0.0:
            verb_maps.append(clean)
    if not verb_maps:
        return "", 0.0, 0.0, 0

    mean_conc = 0.0
    mean_top_share = 0.0
    winner_counts: Dict[str, int] = {}
    mean_share_by_adj: Dict[str, float] = {}
    n_verbs = len(verb_maps)

    for amap in verb_maps:
        total = sum(amap.values())
        if total <= 0.0:
            continue
        top_adj = max(amap.items(), key=lambda kv: kv[1])[0]
        top_share = amap[top_adj] / total
        mean_top_share += top_share
        winner_counts[top_adj] = winner_counts.get(top_adj, 0) + 1
        mean_conc += 1.0 - _normalized_entropy(amap)
        for adj, w in amap.items():
            mean_share_by_adj[adj] = mean_share_by_adj.get(adj, 0.0) + (w / total)

    if n_verbs <= 0:
        return "", 0.0, 0.0, 0
    for adj in list(mean_share_by_adj.keys()):
        mean_share_by_adj[adj] /= n_verbs

    top_adj_global = sorted(
        mean_share_by_adj.keys(),
        key=lambda a: (-winner_counts.get(a, 0), -mean_share_by_adj.get(a, 0.0), a),
    )[0]
    return top_adj_global, (mean_top_share / n_verbs), (mean_conc / n_verbs), len(mean_share_by_adj)



def _compute_global_bias(counts: Dict[str, float]) -> Tuple[str, float, float, int]:
    clean = {str(k): float(v) for k, v in counts.items() if float(v) > 0.0}
    if not clean:
        return "", 0.0, 0.0, 0
    total = sum(clean.values())
    if total <= 0.0:
        return "", 0.0, 0.0, 0
    top_label = max(clean.items(), key=lambda kv: kv[1])[0]
    top_share = clean[top_label] / total
    concentration = 1.0 - _normalized_entropy(clean)
    return top_label, top_share, concentration, len(clean)


def _compute_verb_bias_within_adjectives(
    verb_adj_w: Dict[str, Dict[str, float]],
) -> Tuple[str, float, float, int]:
    adj_maps: Dict[str, Dict[str, float]] = {}
    for verb, amap in verb_adj_w.items():
        for adj, w in amap.items():
            ww = float(w)
            if ww <= 0.0:
                continue
            adj_maps.setdefault(str(adj), {})
            adj_maps[str(adj)][str(verb)] = adj_maps[str(adj)].get(str(verb), 0.0) + ww
    if not adj_maps:
        return "", 0.0, 0.0, 0
    mean_conc = 0.0
    mean_top_share = 0.0
    winner_counts: Dict[str, int] = {}
    mean_share_by_verb: Dict[str, float] = {}
    n_adjs = 0
    for vmap in adj_maps.values():
        clean = {k: float(v) for k, v in vmap.items() if float(v) > 0.0}
        if not clean:
            continue
        n_adjs += 1
        total = sum(clean.values())
        top_verb = max(clean.items(), key=lambda kv: kv[1])[0]
        top_share = clean[top_verb] / total
        mean_top_share += top_share
        winner_counts[top_verb] = winner_counts.get(top_verb, 0) + 1
        mean_conc += 1.0 - _normalized_entropy(clean)
        for verb, w in clean.items():
            mean_share_by_verb[verb] = mean_share_by_verb.get(verb, 0.0) + (w / total)
    if n_adjs <= 0:
        return "", 0.0, 0.0, 0
    for verb in list(mean_share_by_verb.keys()):
        mean_share_by_verb[verb] /= n_adjs
    top_verb_global = sorted(mean_share_by_verb.keys(), key=lambda v: (-winner_counts.get(v, 0), -mean_share_by_verb.get(v, 0.0), v))[0]
    return top_verb_global, (mean_top_share / n_adjs), (mean_conc / n_adjs), len(mean_share_by_verb)


def _role_dominance(verb_score: float, adj_score: float, eps: float = 1e-12) -> float:
    vv = max(0.0, float(verb_score))
    aa = max(0.0, float(adj_score))
    return (vv - aa) / (vv + aa + eps)


def _role_type(verb_score: float, adj_score: float, hi: float = 0.55, lo: float = 0.35) -> str:
    vv = float(verb_score)
    aa = float(adj_score)
    if vv >= hi and aa <= lo:
        return "verb_only"
    if aa >= hi and vv <= lo:
        return "adj_only"
    if vv >= hi and aa >= hi:
        return "mixed"
    return "generic"


def _build_adj_bias_fields(verb_concentration: float, adj_w: Dict[str, float], verb_adj_w: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    top_adj_global, top_adj_share_global, adj_global_concentration, n_adjs_used_global = _compute_global_bias(adj_w)
    top_adj, top_adj_share, adj_concentration, n_adjs_used = _compute_adj_bias_within_verbs(verb_adj_w)
    top_verb_within_adj, top_verb_share_within_adj, verb_within_adj_concentration, n_verbs_used_within_adj = _compute_verb_bias_within_adjectives(verb_adj_w)
    role_dominance_sym = _role_dominance(verb_within_adj_concentration, adj_concentration)
    role_dominance_global = _role_dominance(verb_concentration, adj_global_concentration)
    role_dominance_within = _role_dominance(verb_concentration, adj_concentration)
    return {
        'top_adj': top_adj,
        'top_adj_share': top_adj_share,
        'adj_concentration': adj_concentration,
        'adj_within_verb_concentration': adj_concentration,
        'adj_bias_type': _bias_bucket(adj_concentration, top_adj_share),
        'n_adjs_used': n_adjs_used,
        'top_adj_global': top_adj_global,
        'top_adj_share_global': top_adj_share_global,
        'adj_global_concentration': adj_global_concentration,
        'adj_global_bias_type': _bias_bucket(adj_global_concentration, top_adj_share_global),
        'n_adjs_used_global': n_adjs_used_global,
        'top_verb_within_adj': top_verb_within_adj,
        'top_verb_share_within_adj': top_verb_share_within_adj,
        'verb_within_adj_concentration': verb_within_adj_concentration,
        'n_verbs_used_within_adj': n_verbs_used_within_adj,
        'adj_conditional_gap': adj_concentration - adj_global_concentration,
        'verb_conditional_gap': verb_within_adj_concentration - verb_concentration,
        'role_dominance_sym': role_dominance_sym,
        'role_abs_dominance_sym': abs(role_dominance_sym),
        'role_dominance_global': role_dominance_global,
        'role_dominance_within': role_dominance_within,
        'role_type_sym': _role_type(verb_within_adj_concentration, adj_concentration),
        'role_type_global': _role_type(verb_concentration, adj_global_concentration),
        'bias_strength_within_verb_adj': 0.5 * (verb_concentration + adj_concentration),
        'bias_strength_global_adj': 0.5 * (verb_concentration + adj_global_concentration),
        'bias_strength_mean_adj_modes': 0.5 * (0.5 * (verb_concentration + adj_global_concentration) + 0.5 * (verb_concentration + adj_concentration)),
        'bias_strength': 0.5 * (verb_concentration + adj_concentration),
    }


def _quantile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    qq = min(1.0, max(0.0, float(q)))
    pos = qq * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def export_bias_strength_distribution_and_comparison(
    task: str,
    names: List[str],
    bias_rows: List[Dict[str, object]],
    out_dir: Path,
    bin_width: float,
) -> None:
    dist_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    bw = max(1e-6, min(1.0, float(bin_width)))
    n_bins = int(math.ceil(1.0 / bw))

    for model in names:
        rows = [r for r in bias_rows if r.get("task") == task and r.get("tokenizer") == model]
        if not rows:
            continue

        strengths = []
        verb_cs = []
        adj_cs = []
        verb_types = []
        adj_types = []
        for r in rows:
            vc = _to_float(r.get("verb_concentration"), 0.0)
            ac = _to_float(r.get("adj_concentration"), 0.0)
            vc = min(1.0, max(0.0, vc))
            ac = min(1.0, max(0.0, ac))
            strengths.append(0.5 * (vc + ac))
            verb_cs.append(vc)
            adj_cs.append(ac)
            verb_types.append(str(r.get("verb_bias_type", "")))
            adj_types.append(str(r.get("adj_bias_type", "")))

        total = len(strengths)
        if total == 0:
            continue

        counts = [0 for _ in range(n_bins)]
        for s in strengths:
            bi = min(n_bins - 1, int(s / bw))
            counts[bi] += 1
        for i, c in enumerate(counts):
            lo = i * bw
            hi = min(1.0, (i + 1) * bw)
            dist_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "metric": "bias_strength",
                    "bin_lo": lo,
                    "bin_hi": hi,
                    "count": c,
                    "ratio": c / total,
                }
            )

        mean_strength = sum(strengths) / total
        var_strength = sum((x - mean_strength) ** 2 for x in strengths) / total
        std_strength = math.sqrt(max(0.0, var_strength))
        summary_rows.append(
            {
                "task": task,
                "tokenizer": model,
                "n_ids": total,
                "mean_bias_strength": mean_strength,
                "std_bias_strength": std_strength,
                "median_bias_strength": _quantile(strengths, 0.5),
                "p90_bias_strength": _quantile(strengths, 0.9),
                "mean_verb_concentration": sum(verb_cs) / total,
                "mean_adj_concentration": sum(adj_cs) / total,
                "verb_single_ratio": sum(1 for t in verb_types if t == "single") / total,
                "verb_few_ratio": sum(1 for t in verb_types if t == "few") / total,
                "verb_broad_ratio": sum(1 for t in verb_types if t == "broad") / total,
                "adj_single_ratio": sum(1 for t in adj_types if t == "single") / total,
                "adj_few_ratio": sum(1 for t in adj_types if t == "few") / total,
                "adj_broad_ratio": sum(1 for t in adj_types if t == "broad") / total,
            }
        )

    write_csv(out_dir / f"{task}_bias_strength_distribution.csv", dist_rows)
    write_csv(out_dir / f"{task}_bias_strength_comparison.csv", summary_rows)


def export_role_disentanglement_comparison(
    task: str,
    names: List[str],
    bias_rows: List[Dict[str, object]],
    out_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for model in names:
        sub = [r for r in bias_rows if r.get("task") == task and r.get("tokenizer") == model]
        if not sub:
            continue
        ws = [max(0.0, _to_float(r.get("importance_score"), 1.0)) for r in sub]
        if sum(ws) <= 0.0:
            ws = [1.0 for _ in sub]
        total_w = sum(ws)
        def _wavg(key: str) -> float:
            vals = [_to_float(r.get(key), float("nan")) for r in sub]
            num = 0.0
            den = 0.0
            for v, w in zip(vals, ws):
                if math.isnan(v):
                    continue
                num += w * v
                den += w
            return num / den if den > 0.0 else float("nan")
        def _wratio(role: str, key: str = "role_type_sym") -> float:
            num = 0.0
            den = 0.0
            for r, w in zip(sub, ws):
                den += w
                if str(r.get(key, "")) == role:
                    num += w
            return num / den if den > 0.0 else float("nan")
        rows.append({
            "task": task,
            "tokenizer": model,
            "n_ids": len(sub),
            "weighted_mean_role_dominance_sym": _wavg("role_dominance_sym"),
            "weighted_mean_abs_role_dominance_sym": _wavg("role_abs_dominance_sym"),
            "weighted_mean_role_dominance_global": _wavg("role_dominance_global"),
            "weighted_mean_role_dominance_within": _wavg("role_dominance_within"),
            "weighted_mean_adj_conditional_gap": _wavg("adj_conditional_gap"),
            "weighted_mean_verb_conditional_gap": _wavg("verb_conditional_gap"),
            "weighted_verb_only_ratio": _wratio("verb_only"),
            "weighted_adj_only_ratio": _wratio("adj_only"),
            "weighted_mixed_ratio": _wratio("mixed"),
            "weighted_generic_ratio": _wratio("generic"),
        })
    write_csv(out_dir / f"{task}_role_disentanglement_comparison.csv", rows)


def _top_rows_for_model(rows: List[Dict[str, str]], top_k: int) -> List[Dict[str, object]]:
    picked = sorted(
        rows,
        key=lambda r: (-_to_float(r.get("importance_score"), 0.0), int(r.get("item_id", 0))),
    )[: max(1, top_k)]
    out: List[Dict[str, object]] = []
    for i, r in enumerate(picked, start=1):
        out.append(
            {
                "rank": i,
                "item_id": int(r["item_id"]),
                "item_name": r.get("item_name", r["item_id"]),
                "importance_score": _to_float(r.get("importance_score"), 0.0),
            }
        )
    return out


def plot_top_id_importance_per_model(
    task: str,
    rows_by_model: Dict[str, List[Dict[str, str]]],
    names: List[str],
    top_k: int,
    out_dir: Path,
) -> List[Dict[str, object]]:
    merged_rows: List[Dict[str, object]] = []
    models = [n for n in names if n in rows_by_model]
    if not models:
        print(f"[warn] no rows for top-id importance: task={task}")
        return merged_rows

    for model in models:
        top_rows = _top_rows_for_model(rows_by_model[model], top_k)
        for r in top_rows:
            rr = dict(r)
            rr["task"] = task
            rr["tokenizer"] = model
            merged_rows.append(rr)

        if not HAS_MATPLOTLIB:
            continue
        fig_h = max(4.5, 0.28 * len(top_rows) + 1.8)
        fig, ax = plt.subplots(1, 1, figsize=(8.4, fig_h), constrained_layout=True)
        labels = [str(r["item_id"]) for r in top_rows]
        values = [float(r["importance_score"]) for r in top_rows]
        ypos = list(range(len(top_rows)))
        ax.barh(ypos, values)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel("importance_score")
        ax.set_ylabel("token id")
        ax.set_title(f"{task} | {model} | top-{len(top_rows)} importance IDs")
        ax.grid(axis="x", alpha=0.25)
        fig.savefig(out_dir / f"{task}_{_sanitize_name(model)}_top_id_importance.png", dpi=180)
        plt.close(fig)

    if not HAS_MATPLOTLIB:
        print(f"[warn] matplotlib not found: skip top-id bar plots ({task})")
    return merged_rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    keys_seen = set()
    for r in rows:
        for k in r.keys():
            if k not in keys_seen:
                keys_seen.add(k)
                fieldnames.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_base_metrics_csv(
    summaries: Dict[str, Dict[str, Dict[str, object]]],
    names: List[str],
    out_path: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for name in names:
        if name in summaries["actionrec"]:
            b = summaries["actionrec"][name]["base_metrics"]
            rows.append(
                {
                    "task": "actionrec",
                    "tokenizer": name,
                    "acc": _to_float(b.get("acc")),
                    "macro_f1": _to_float(b.get("macro_f1")),
                    "mean_true_prob": _to_float(b.get("mean_true_prob")),
                }
            )
        if name in summaries["retrieval"]:
            b = summaries["retrieval"][name]["base_metrics"]
            t2m_r1 = _to_float(b["t2m"].get("R@1"))
            m2t_r1 = _to_float(b["m2t"].get("R@1"))
            rows.append(
                {
                    "task": "retrieval",
                    "tokenizer": name,
                    "t2m_R@1": t2m_r1,
                    "m2t_R@1": m2t_r1,
                    "mean_R@1": (t2m_r1 + m2t_r1) / 2.0,
                    "t2m_R@5": _to_float(b["t2m"].get("R@5")),
                    "m2t_R@5": _to_float(b["m2t"].get("R@5")),
                    "t2m_R@10": _to_float(b["t2m"].get("R@10")),
                    "m2t_R@10": _to_float(b["m2t"].get("R@10")),
                }
            )
    write_csv(out_path, rows)


def _build_sample_id_sets(
    sample_rows: List[Dict[str, str]],
    mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
) -> List[Dict[str, object]]:
    by_sample: Dict[str, List[Dict[str, str]]] = {}
    for r in sample_rows:
        mid = str(r.get("mid", "")).strip()
        if not mid:
            continue
        by_sample.setdefault(mid, []).append(r)

    use_top_p = 0.0 < float(sample_top_p) < 1.0
    out_rows: List[Dict[str, object]] = []

    def _build_scored(rows: List[Dict[str, str]]) -> List[Tuple[int, str, str, float]]:
        scored: List[Tuple[int, str, str, float]] = []
        for r in rows:
            iid = int(_to_float(r.get("item_id"), 0.0))
            w = _to_float(r.get("abs_attr_norm"), abs(_to_float(r.get("signed_attr_norm"), 0.0)))
            if w <= 0.0:
                continue
            verb, adj = _extract_verb_adj_from_row(r)
            scored.append((iid, verb, adj, w))
        return sorted(scored, key=lambda x: (-x[3], x[0]))

    def _apply_top_p(scored: List[Tuple[int, str, str, float]]) -> List[Tuple[int, str, str, float]]:
        if not use_top_p:
            return list(scored)
        p = min(1.0, max(1e-8, float(sample_top_p)))
        cum = 0.0
        selected: List[Tuple[int, str, str, float]] = []
        for t in scored:
            selected.append(t)
            cum += float(t[3])
            if cum >= p:
                break
        return selected

    for mid, rows in by_sample.items():
        scored = _build_scored(rows)
        if not scored:
            continue
        scored = _apply_top_p(scored)
        if str(mode).strip() == "sample_topk_count" and int(sample_top_k_ids) > 0:
            scored = scored[: int(sample_top_k_ids)]
        ids = sorted({int(t[0]) for t in scored})
        if not ids:
            continue
        verb, adj = _extract_verb_adj_from_row(rows[0])
        out_rows.append(
            {
                "mid": mid,
                "verb": verb,
                "adjective": adj,
                "n_ids": len(ids),
                "selected_ids_json": json.dumps(ids, ensure_ascii=False),
            }
        )
    return out_rows


def _mean_pairwise_jaccard(id_sets: List[set]) -> float:
    if len(id_sets) <= 1:
        return 1.0 if id_sets else float("nan")
    vals: List[float] = []
    for i in range(len(id_sets)):
        for j in range(i + 1, len(id_sets)):
            a = id_sets[i]
            b = id_sets[j]
            denom = len(a | b)
            vals.append((len(a & b) / denom) if denom > 0 else 0.0)
    return (sum(vals) / len(vals)) if vals else float("nan")


def _extract_conditional_id_set_motifs(
    sample_set_rows: List[Dict[str, object]],
    context_key: str,
    target_key: str,
    min_support: float,
    min_samples: int,
    top_k_ids: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    by_context: Dict[str, List[Dict[str, object]]] = {}
    for r in sample_set_rows:
        context = str(r.get(context_key, "")).strip()
        target = str(r.get(target_key, "")).strip()
        if not context or not target:
            continue
        grouped.setdefault((context, target), []).append(r)
        by_context.setdefault(context, []).append(r)

    summary_rows: List[Dict[str, object]] = []
    item_rows: List[Dict[str, object]] = []

    for (context, target), rows in sorted(grouped.items()):
        if len(rows) < int(min_samples):
            continue
        cond_sets = [set(json.loads(str(r.get("selected_ids_json", "[]")))) for r in rows]
        cond_sets = [s for s in cond_sets if s]
        if len(cond_sets) < int(min_samples):
            continue
        other_rows = [r for r in by_context.get(context, []) if str(r.get(target_key, "")).strip() != target]
        other_sets = [set(json.loads(str(r.get("selected_ids_json", "[]")))) for r in other_rows]
        other_sets = [s for s in other_sets if s]

        cond_counts: Dict[int, int] = {}
        other_counts: Dict[int, int] = {}
        for s in cond_sets:
            for iid in s:
                cond_counts[int(iid)] = cond_counts.get(int(iid), 0) + 1
        for s in other_sets:
            for iid in s:
                other_counts[int(iid)] = other_counts.get(int(iid), 0) + 1

        denom_cond = float(len(cond_sets))
        denom_other = float(len(other_sets))
        candidates: List[Dict[str, object]] = []
        for iid in sorted(set(cond_counts.keys()) | set(other_counts.keys())):
            support_cond = cond_counts.get(iid, 0) / denom_cond if denom_cond > 0 else 0.0
            support_other = other_counts.get(iid, 0) / denom_other if denom_other > 0 else 0.0
            support_diff = support_cond - support_other
            support_lift = (support_cond / support_other) if support_other > 0 else (float("inf") if support_cond > 0 else float("nan"))
            motif_score = max(0.0, support_diff) * support_cond
            candidates.append({
                "context_label": context,
                "target_label": target,
                "item_id": int(iid),
                "support_cond": support_cond,
                "support_other": support_other,
                "support_diff_vs_other": support_diff,
                "support_lift_vs_other": support_lift,
                "motif_score": motif_score,
            })

        selected = [
            r for r in candidates
            if float(r["support_cond"]) >= float(min_support) and float(r["support_diff_vs_other"]) > 0.0
        ]
        selected.sort(key=lambda r: (-float(r["motif_score"]), -float(r["support_diff_vs_other"]), -float(r["support_cond"]), int(r["item_id"])))
        if int(top_k_ids) > 0:
            selected = selected[: int(top_k_ids)]

        rep_ids = [int(r["item_id"]) for r in selected]
        summary_rows.append({
            "context_label": context,
            "target_label": target,
            "n_samples": len(cond_sets),
            "n_other_samples_same_context": len(other_sets),
            "group_mean_pairwise_jaccard": _mean_pairwise_jaccard(cond_sets),
            "n_motif_ids": len(rep_ids),
            "motif_ids_json": json.dumps(rep_ids, ensure_ascii=False),
        })
        for rank, row in enumerate(selected, start=1):
            rr = dict(row)
            rr["rank"] = rank
            item_rows.append(rr)

    return summary_rows, item_rows


def export_conditional_id_set_motifs(
    task: str,
    names: List[str],
    attr_sample_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_sample_aggregation_mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
    motif_min_support: float,
    motif_min_samples: int,
    motif_top_k_ids: int,
    out_dir: Path,
) -> None:
    summary_rows: List[Dict[str, object]] = []
    item_rows: List[Dict[str, object]] = []
    sample_set_rows_out: List[Dict[str, object]] = []

    for model in names:
        sample_rows = attr_sample_rows_by_model.get(model, [])
        if not sample_rows:
            continue
        sample_set_rows = _build_sample_id_sets(
            sample_rows,
            attr_sample_aggregation_mode,
            sample_top_k_ids,
            sample_top_p,
        )
        for row in sample_set_rows:
            rr = dict(row)
            rr["task"] = task
            rr["tokenizer"] = model
            sample_set_rows_out.append(rr)
        for motif_mode, context_key, target_key in [
            ("adj_given_verb", "verb", "adjective"),
            ("verb_given_adj", "adjective", "verb"),
        ]:
            s_rows, i_rows = _extract_conditional_id_set_motifs(
                sample_set_rows,
                context_key,
                target_key,
                motif_min_support,
                motif_min_samples,
                motif_top_k_ids,
            )
            for row in s_rows:
                rr = dict(row)
                rr["task"] = task
                rr["tokenizer"] = model
                rr["motif_mode"] = motif_mode
                summary_rows.append(rr)
            for row in i_rows:
                rr = dict(row)
                rr["task"] = task
                rr["tokenizer"] = model
                rr["motif_mode"] = motif_mode
                item_rows.append(rr)

    write_csv(out_dir / f"{task}_conditional_id_set_motif_summary.csv", summary_rows)
    write_csv(out_dir / f"{task}_conditional_id_set_motif_items.csv", item_rows)
    write_csv(out_dir / f"{task}_conditional_id_set_sample_sets.csv", sample_set_rows_out)


def plot_conditional_id_set_support_bars(
    task: str,
    names: List[str],
    out_dir: Path,
    max_ids_per_group: int = 20,
) -> None:
    summary_path = out_dir / f"{task}_conditional_id_set_motif_summary.csv"
    sample_sets_path = out_dir / f"{task}_conditional_id_set_sample_sets.csv"
    if not summary_path.exists() or not sample_sets_path.exists():
        print(f"[warn] missing conditional motif CSVs for support scatter: task={task}")
        return

    summary_rows = _load_csv_rows(summary_path)
    sample_set_rows = _load_csv_rows(sample_sets_path)
    if not summary_rows or not sample_set_rows:
        print(f"[warn] no conditional motif rows for support scatter: task={task}")
        return

    by_model: Dict[str, List[Dict[str, str]]] = {}
    for r in sample_set_rows:
        model = str(r.get("tokenizer", "")).strip()
        if not model:
            continue
        by_model.setdefault(model, []).append(r)

    def _parse_id_set(s: str) -> set:
        try:
            vals = json.loads(str(s))
            return {int(v) for v in vals}
        except (json.JSONDecodeError, TypeError, ValueError):
            return set()

    def _compute_set_support(subset: set, sets: List[set]) -> float:
        if not sets:
            return 0.0
        n = 0
        for ss in sets:
            if subset.issubset(ss):
                n += 1
        return n / float(len(sets))

    def _greedy_max_diff(candidates: set, cond_sets: List[set], other_sets: List[set]) -> Tuple[set, float, float]:
        if not candidates:
            return set(), 0.0, 0.0

        def _score(subset: set) -> Tuple[float, float, float]:
            s_cond = _compute_set_support(subset, cond_sets)
            s_other = _compute_set_support(subset, other_sets)
            return s_cond - s_other, s_cond, s_other

        cand_sorted = sorted(int(i) for i in candidates)
        # Start from the best single ID.
        best_subset: set = {cand_sorted[0]}
        best_diff, best_cond, best_other = _score(best_subset)
        for iid in cand_sorted[1:]:
            d, c, o = _score({iid})
            if (d > best_diff) or (abs(d - best_diff) <= 1e-12 and (c > best_cond or (abs(c - best_cond) <= 1e-12 and iid < min(best_subset)))):
                best_subset = {iid}
                best_diff, best_cond, best_other = d, c, o

        # Forward greedy add if it improves diff.
        improved = True
        while improved:
            improved = False
            remain = [iid for iid in cand_sorted if iid not in best_subset]
            pick_iid = None
            pick_diff = best_diff
            pick_cond = best_cond
            pick_other = best_other
            for iid in remain:
                trial = set(best_subset)
                trial.add(iid)
                d, c, o = _score(trial)
                if (d > pick_diff + 1e-12) or (
                    abs(d - pick_diff) <= 1e-12 and (c > pick_cond + 1e-12 or (abs(c - pick_cond) <= 1e-12 and iid < (pick_iid if pick_iid is not None else 10**18)))
                ):
                    pick_iid = iid
                    pick_diff = d
                    pick_cond = c
                    pick_other = o
            if pick_iid is not None:
                best_subset.add(pick_iid)
                best_diff = pick_diff
                best_cond = pick_cond
                best_other = pick_other
                improved = True

        return best_subset, best_cond, best_other

    point_rows: List[Dict[str, object]] = []
    mode_to_keys = {
        "adj_given_verb": ("verb", "adjective"),
        "verb_given_adj": ("adjective", "verb"),
    }
    for r in summary_rows:
        model = str(r.get("tokenizer", "")).strip()
        mode = str(r.get("motif_mode", "")).strip()
        context = str(r.get("context_label", "")).strip()
        target = str(r.get("target_label", "")).strip()
        if not model or model not in names or mode not in mode_to_keys or not context or not target:
            continue
        candidate_ids = _parse_id_set(str(r.get("motif_ids_json", "[]")))
        if not candidate_ids:
            continue
        context_key, target_key = mode_to_keys[mode]
        sample_rows_model = by_model.get(model, [])
        cond_rows = [
            rr for rr in sample_rows_model
            if str(rr.get(context_key, "")).strip() == context and str(rr.get(target_key, "")).strip() == target
        ]
        other_rows = [
            rr for rr in sample_rows_model
            if str(rr.get(context_key, "")).strip() == context and str(rr.get(target_key, "")).strip() != target
        ]
        if not cond_rows:
            continue

        cond_sets = [_parse_id_set(str(rr.get("selected_ids_json", "[]"))) for rr in cond_rows]
        other_sets = [_parse_id_set(str(rr.get("selected_ids_json", "[]"))) for rr in other_rows]
        motif_ids, support_cond_set, support_other_set = _greedy_max_diff(candidate_ids, cond_sets, other_sets)
        if not motif_ids:
            continue
        point_rows.append(
            {
                "task": task,
                "tokenizer": model,
                "motif_mode": mode,
                "context_label": context,
                "target_label": target,
                "n_candidate_ids": len(candidate_ids),
                "n_motif_ids": len(motif_ids),
                "motif_ids_json": json.dumps(sorted(motif_ids), ensure_ascii=False),
                "n_cond_rows": len(cond_sets),
                "n_other_rows": len(other_sets),
                "set_support_cond": support_cond_set,
                "set_support_other": support_other_set,
                "set_support_diff": support_cond_set - support_other_set,
            }
        )

    write_csv(out_dir / f"{task}_conditional_id_set_motif_set_support_points.csv", point_rows)
    if not point_rows:
        print(f"[warn] no valid motif-id sets for support scatter: task={task}")
        return
    if not HAS_MATPLOTLIB:
        print(f"[warn] matplotlib not found: skip conditional motif support scatter PNG ({task})")
        return

    for mode in ("adj_given_verb", "verb_given_adj"):
        rows_mode = [r for r in point_rows if str(r.get("motif_mode", "")) == mode]
        if not rows_mode:
            continue
        for model in names:
            rows_model = [r for r in rows_mode if str(r.get("tokenizer", "")) == model]
            if not rows_model:
                continue
            fig, ax = plt.subplots(1, 1, figsize=(8.0, 7.0), constrained_layout=True)
            if mode == "adj_given_verb":
                color_label_name = "verb"
                marker_label_name = "adjective"
            else:
                color_label_name = "adjective"
                marker_label_name = "verb"
            color_values = sorted({str(r.get("context_label", "")).strip() for r in rows_model})
            marker_values = sorted({str(r.get("target_label", "")).strip() for r in rows_model})
            cmap = plt.get_cmap("tab20")
            color_by_value = {lb: cmap(i % 20) for i, lb in enumerate(color_values)}
            color_idx_by_value = {lb: i for i, lb in enumerate(color_values)}
            markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
            marker_by_value = {lb: markers[i % len(markers)] for i, lb in enumerate(marker_values)}
            marker_idx_by_value = {lb: i for i, lb in enumerate(marker_values)}
            for row in rows_model:
                x = _to_float(row.get("set_support_other"), 0.0)
                y = _to_float(row.get("set_support_cond"), 0.0)
                color_value = str(row.get("context_label", "")).strip()
                marker_value = str(row.get("target_label", "")).strip()
                # Small deterministic jitter to avoid exact overlap.
                dx = ((color_idx_by_value.get(color_value, 0) % 7) - 3) * 0.0025
                dy = ((marker_idx_by_value.get(marker_value, 0) % 7) - 3) * 0.0025
                x_plot = min(1.0, max(0.0, x + dx))
                y_plot = min(1.0, max(0.0, y + dy))
                ax.scatter(
                    [x_plot],
                    [y_plot],
                    s=30 + 6 * max(1.0, _to_float(row.get("n_motif_ids"), 1.0)),
                    marker=marker_by_value.get(marker_value, "o"),
                    color=color_by_value.get(color_value, "#4C78A8"),
                    alpha=0.8,
                    edgecolors="none",
                )

            ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, color="#666666", alpha=0.8)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("set_support_other")
            ax.set_ylabel("set_support_cond")
            ax.set_title(f"{task} | {model} | {mode} | Motif ID-set support")
            ax.grid(alpha=0.25)

            color_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markerfacecolor=color_by_value[lb],
                    markersize=7,
                    label=lb,
                )
                for lb in color_values
            ]
            marker_handles = [
                Line2D(
                    [0],
                    [0],
                    marker=marker_by_value[lb],
                    color="#444444",
                    linestyle="none",
                    markersize=7,
                    label=lb,
                )
                for lb in marker_values
            ]
            leg1 = ax.legend(
                handles=color_handles,
                title=f"Color -> {color_label_name}",
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                fontsize=7,
                title_fontsize=8,
                frameon=True,
            )
            ax.add_artist(leg1)
            ax.legend(
                handles=marker_handles,
                title=f"Marker -> {marker_label_name}",
                loc="lower left",
                bbox_to_anchor=(1.02, 0.0),
                fontsize=7,
                title_fontsize=8,
                frameon=True,
            )
            fig.savefig(
                out_dir / f"{task}_{_sanitize_name(model)}_conditional_id_set_motif_set_support_scatter_{mode}.png",
                dpi=180,
            )
            plt.close(fig)


def _group_rows_by_class(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    by_class: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        class_name = r.get("class_name", "")
        by_class.setdefault(class_name, []).append(r)
    return by_class


def export_class_id_association_per_model(
    task: str,
    class_rows_by_model: Dict[str, List[Dict[str, str]]],
    names: List[str],
    top_k_ids_per_class: int,
    out_dir: Path,
) -> List[Dict[str, object]]:
    merged: List[Dict[str, object]] = []
    models = [n for n in names if n in class_rows_by_model]
    if not models:
        print(f"[warn] no class rows for class-id association: task={task}")
        return merged

    for model in models:
        rows = class_rows_by_model[model]
        by_class = _group_rows_by_class(rows)
        for class_name, crow_list in by_class.items():
            class_id = int(crow_list[0].get("class_id", -1))
            valid = [r for r in crow_list if _to_float(r.get("samples_with_item_in_class"), 0.0) > 0.0]
            if not valid:
                continue
            picked = sorted(
                valid,
                key=lambda r: (-_to_float(r.get("chi2_2x2"), 0.0), -_to_float(r.get("sample_freq_given_class"), 0.0)),
            )[: max(1, top_k_ids_per_class)]
            for i, r in enumerate(picked, start=1):
                merged.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "class_id": class_id,
                        "class_name": class_name,
                        "rank": i,
                        "item_id": int(r.get("item_id", 0)),
                        "item_name": r.get("item_name", r.get("item_id", "")),
                        "chi2_2x2": _to_float(r.get("chi2_2x2"), 0.0),
                        "pmi_bits": _to_float(r.get("pmi_bits"), float("-inf")),
                        "sample_freq_given_class": _to_float(r.get("sample_freq_given_class"), 0.0),
                        "samples_with_item_in_class": int(_to_float(r.get("samples_with_item_in_class"), 0.0)),
                    }
                )
    write_csv(out_dir / f"{task}_class_id_association_per_model.csv", merged)
    return merged


def plot_class_id_association_heatmap(
    task: str,
    class_assoc_rows: List[Dict[str, object]],
    names: List[str],
    top_k_classes: int,
    top_k_ids_per_class: int,
    out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print(f"[warn] matplotlib not found: skip class-id heatmaps ({task})")
        return

    for model in names:
        rows = [r for r in class_assoc_rows if r["tokenizer"] == model and r["task"] == task]
        if not rows:
            continue
        class_best: Dict[str, float] = {}
        for r in rows:
            cn = str(r["class_name"])
            class_best[cn] = max(class_best.get(cn, 0.0), _to_float(r["chi2_2x2"], 0.0))
        classes = sorted(class_best.keys(), key=lambda c: (-class_best[c], c))[: max(1, top_k_classes)]

        mat: List[List[float]] = []
        ann: List[List[str]] = []
        for cn in classes:
            crows = [r for r in rows if r["class_name"] == cn]
            crows = sorted(crows, key=lambda r: int(r["rank"]))[: max(1, top_k_ids_per_class)]
            vals = [_to_float(r["chi2_2x2"], 0.0) for r in crows]
            ids = [str(r["item_id"]) for r in crows]
            while len(vals) < top_k_ids_per_class:
                vals.append(0.0)
                ids.append("-")
            mat.append(vals)
            ann.append(ids)

        fig_h = max(5.0, 0.32 * len(classes) + 1.6)
        fig_w = max(7.0, 1.15 * top_k_ids_per_class + 2.5)
        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)
        im = ax.imshow(mat, aspect="auto", cmap="Blues")
        ax.set_title(f"{task} | {model} | class-ID association (chi2)")
        ax.set_xlabel("ID rank within class")
        ax.set_ylabel("class")
        ax.set_xticks(list(range(top_k_ids_per_class)))
        ax.set_xticklabels([str(i + 1) for i in range(top_k_ids_per_class)])
        ax.set_yticks(list(range(len(classes))))
        ax.set_yticklabels(classes)
        for yi in range(len(classes)):
            for xi in range(top_k_ids_per_class):
                ax.text(xi, yi, ann[yi][xi], ha="center", va="center", fontsize=7)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("chi2_2x2")
        fig.savefig(out_dir / f"{task}_{_sanitize_name(model)}_class_id_association_heatmap.png", dpi=180)
        plt.close(fig)


def export_id_label_bias_per_model(
    task: str,
    names: List[str],
    perturb_rows_by_model: Dict[str, List[Dict[str, str]]],
    class_rows_by_model: Dict[str, List[Dict[str, str]]],
    top_k_ids: int,
    out_dir: Path,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    models = [n for n in names if n in perturb_rows_by_model and n in class_rows_by_model]
    if not models:
        print(f"[warn] no rows for id-label bias: task={task}")
        return out

    for model in models:
        top_ids = _top_rows_for_model(perturb_rows_by_model[model], top_k_ids)
        class_rows = class_rows_by_model[model]
        by_id: Dict[int, List[Dict[str, str]]] = {}
        for r in class_rows:
            iid = int(r.get("item_id", 0))
            by_id.setdefault(iid, []).append(r)

        for rtop in top_ids:
            iid = int(rtop["item_id"])
            imp = max(0.0, _to_float(rtop["importance_score"], 0.0))
            rows = by_id.get(iid, [])
            verb_w: Dict[str, float] = {}
            adj_w: Dict[str, float] = {}
            adj_by_verb: Dict[str, Dict[str, float]] = {}
            total_w = 0.0
            for rr in rows:
                if _to_float(rr.get("samples_with_item_in_class"), 0.0) <= 0.0:
                    continue
                class_name = rr.get("class_name", "")
                verb, adj = _split_class_name(class_name)
                w = (
                    imp
                    * _to_float(rr.get("chi2_2x2"), 0.0)
                    # * _to_float(rr.get("sample_freq_given_class"), 0.0)
                )
                if w <= 0.0:
                    continue
                verb_w[verb] = verb_w.get(verb, 0.0) + w
                adj_w[adj] = adj_w.get(adj, 0.0) + w
                adj_by_verb.setdefault(verb, {})
                adj_by_verb[verb][adj] = adj_by_verb[verb].get(adj, 0.0) + w
                total_w += w

            if total_w <= 0.0:
                continue

            top_verb = max(verb_w.items(), key=lambda kv: kv[1])[0]
            top_verb_share = verb_w[top_verb] / max(total_w, 1e-12)
            verb_entropy_n = _normalized_entropy(verb_w)
            verb_concentration = 1.0 - verb_entropy_n
            top_adj, top_adj_share, adj_concentration, n_adjs_used = _compute_adj_bias_within_verbs(adj_by_verb)
            bias_strength = 0.5 * (verb_concentration + adj_concentration)
            role_fields = _build_adj_bias_fields(verb_concentration, adj_w, adj_by_verb)

            out.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "item_id": iid,
                    "importance_score": _to_float(rtop["importance_score"], 0.0),
                    "top_verb": top_verb,
                    "top_verb_share": top_verb_share,
                    "verb_concentration": verb_concentration,
                    "verb_bias_type": _bias_bucket(verb_concentration, top_verb_share),
                    "n_verbs_used": len(verb_w),
                    "top_adj": top_adj,
                    "top_adj_share": top_adj_share,
                    "adj_concentration": adj_concentration,
                    "adj_bias_type": _bias_bucket(adj_concentration, top_adj_share),
                    "n_adjs_used": n_adjs_used,
                    "bias_strength": bias_strength,
                    **role_fields,
                }
            )

    out = sorted(
        out,
        key=lambda r: (
            str(r.get("task", "")),
            str(r.get("tokenizer", "")),
            -_to_float(r.get("bias_strength"), 0.0),
            int(r.get("item_id", 0)),
        ),
    )
    write_csv(out_dir / f"{task}_id_label_bias_per_model.csv", out)
    return out


def export_id_label_bias_from_attr_per_model(
    task: str,
    names: List[str],
    attr_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_sample_rows_by_model: Dict[str, List[Dict[str, str]]],
    top_k_ids: int,
    attr_sample_aggregation_mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
    out_dir: Path,
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    models = [n for n in names if n in attr_rows_by_model and n in attr_sample_rows_by_model]
    if not models:
        print(f"[warn] no attribution rows for id-label bias map: task={task}")
        return out

    for model in models:
        sample_rows = attr_sample_rows_by_model.get(model, [])
        total_by_id, verb_by_id, adj_by_id, verb_adj_by_id = _aggregate_attr_samples_by_mode(
            sample_rows=sample_rows,
            mode=attr_sample_aggregation_mode,
            sample_top_k_ids=sample_top_k_ids,
            sample_top_p=sample_top_p,
        )

        top_attr = sorted(
            attr_rows_by_model[model],
            key=lambda r: (-abs(_to_float(r.get("signed_attr_sum"), 0.0)), int(r.get("item_id", 0))),
        )
        if top_k_ids > 0:
            top_attr = top_attr[:top_k_ids]
        if not top_attr:
            top_attr = [
                {"item_id": iid, "importance_score": total}
                for iid, total in total_by_id.items()
            ]
            top_attr = sorted(top_attr, key=lambda r: -_to_float(r.get("importance_score"), 0.0))
            if top_k_ids > 0:
                top_attr = top_attr[:top_k_ids]

        for rtop in top_attr:
            iid = int(rtop["item_id"])
            imp = _to_float(rtop.get("importance_score"), float("nan"))
            if math.isnan(imp):
                imp = total_by_id.get(iid, abs(_to_float(rtop.get("signed_attr_sum"), 0.0)))
            verb_w = dict(verb_by_id.get(iid, {}))
            adj_w = dict(adj_by_id.get(iid, {}))
            verb_adj_w = dict(verb_adj_by_id.get(iid, {}))
            total_w = sum(verb_w.values())

            if total_w <= 0.0:
                continue

            top_verb = max(verb_w.items(), key=lambda kv: kv[1])[0]
            top_verb_share = verb_w[top_verb] / max(total_w, 1e-12)
            verb_concentration = 1.0 - _normalized_entropy(verb_w)
            top_adj, top_adj_share, adj_concentration, n_adjs_used = _compute_adj_bias_within_verbs(
                verb_adj_w
            )
            bias_strength = 0.5 * (verb_concentration + adj_concentration)
            role_fields = _build_adj_bias_fields(verb_concentration, adj_w, verb_adj_w)

            out.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "item_id": iid,
                    "importance_score": imp,
                    "top_verb": top_verb,
                    "top_verb_share": top_verb_share,
                    "verb_concentration": verb_concentration,
                    "verb_bias_type": _bias_bucket(verb_concentration, top_verb_share),
                    "n_verbs_used": len(verb_w),
                    "top_adj": top_adj,
                    "top_adj_share": top_adj_share,
                    "adj_concentration": adj_concentration,
                    "adj_bias_type": _bias_bucket(adj_concentration, top_adj_share),
                    "n_adjs_used": n_adjs_used,
                    "bias_strength": bias_strength,
                    "weight_source": (
                        f"id_attribution_per_sample:{attr_sample_aggregation_mode}"
                        + (f"(topk={int(sample_top_k_ids)})" if str(attr_sample_aggregation_mode) == "sample_topk_count" else "")
                        + (
                            f"(p={float(sample_top_p):.3f})"
                            if (0.0 < float(sample_top_p) < 1.0 and str(attr_sample_aggregation_mode) in {"sample_top_p_count", "abs_sum"})
                            else ""
                        )
                    ),
                    **role_fields,
                }
            )

    out = sorted(
        out,
        key=lambda r: (
            str(r.get("task", "")),
            str(r.get("tokenizer", "")),
            -_to_float(r.get("bias_strength"), 0.0),
            int(r.get("item_id", 0)),
        ),
    )
    write_csv(out_dir / f"{task}_id_label_bias_per_model.csv", out)
    return out


def export_model_label_bias_from_attr_per_model(
    task: str,
    names: List[str],
    attr_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_sample_rows_by_model: Dict[str, List[Dict[str, str]]],
    top_k_ids: int,
    attr_sample_aggregation_mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
    out_dir: Path,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    models = [n for n in names if n in attr_rows_by_model and n in attr_sample_rows_by_model]
    if not models:
        print(f"[warn] no attribution rows for model-label bias summary: task={task}")
        write_csv(out_dir / f"{task}_model_label_bias_from_attr_summary.csv", summary_rows)
        write_csv(out_dir / f"{task}_model_label_bias_from_attr_detail.csv", detail_rows)
        return summary_rows, detail_rows

    for model in models:
        sample_rows = attr_sample_rows_by_model.get(model, [])
        total_by_id, verb_by_id, adj_by_id, verb_adj_by_id = _aggregate_attr_samples_by_mode(
            sample_rows=sample_rows,
            mode=attr_sample_aggregation_mode,
            sample_top_k_ids=sample_top_k_ids,
            sample_top_p=sample_top_p,
        )

        top_attr = sorted(
            attr_rows_by_model[model],
            key=lambda r: (-abs(_to_float(r.get("signed_attr_sum"), 0.0)), int(r.get("item_id", 0))),
        )
        if top_k_ids > 0:
            top_attr = top_attr[:top_k_ids]
        if not top_attr:
            top_attr = [{"item_id": iid, "importance_score": total} for iid, total in total_by_id.items()]
            top_attr = sorted(top_attr, key=lambda r: -_to_float(r.get("importance_score"), 0.0))
            if top_k_ids > 0:
                top_attr = top_attr[:top_k_ids]

        selected_ids = [int(_to_float(r.get("item_id"), 0.0)) for r in top_attr]
        verb_w: Dict[str, float] = {}
        adj_w: Dict[str, float] = {}
        adj_by_verb_all: Dict[str, Dict[str, float]] = {}
        n_effective_ids = 0
        for iid in selected_ids:
            vmap = verb_by_id.get(iid, {})
            amap = adj_by_id.get(iid, {})
            if sum(vmap.values()) <= 0.0 or sum(amap.values()) <= 0.0:
                continue
            n_effective_ids += 1
            for k, v in vmap.items():
                if v > 0.0:
                    verb_w[k] = verb_w.get(k, 0.0) + v
            for k, v in amap.items():
                if v > 0.0:
                    adj_w[k] = adj_w.get(k, 0.0) + v
            for verb, amap2 in verb_adj_by_id.get(iid, {}).items():
                adj_by_verb_all.setdefault(verb, {})
                for adj, w in amap2.items():
                    if w > 0.0:
                        adj_by_verb_all[verb][adj] = adj_by_verb_all[verb].get(adj, 0.0) + w

        verb_total = sum(verb_w.values())
        adj_total = sum(adj_w.values())
        if verb_total <= 0.0 or adj_total <= 0.0:
            continue

        top_verb = max(verb_w.items(), key=lambda kv: kv[1])[0]
        top_verb_share = verb_w[top_verb] / verb_total
        verb_concentration = 1.0 - _normalized_entropy(verb_w)
        adj_fields = _build_adj_bias_fields(verb_concentration, adj_w, adj_by_verb_all)

        weight_source = (
            f"id_attribution_per_sample:{attr_sample_aggregation_mode}"
            + (f"(topk={int(sample_top_k_ids)})" if str(attr_sample_aggregation_mode) == "sample_topk_count" else "")
            + (
                f"(p={float(sample_top_p):.3f})"
                if (0.0 < float(sample_top_p) < 1.0 and str(attr_sample_aggregation_mode) in {"sample_top_p_count", "abs_sum"})
                else ""
            )
        )
        row = {
            "task": task,
            "tokenizer": model,
            "weight_source": weight_source,
            "n_selected_ids": len(selected_ids),
            "n_effective_ids": n_effective_ids,
            "top_verb": top_verb,
            "top_verb_share": top_verb_share,
            "verb_concentration": verb_concentration,
            "n_verbs_used": len(verb_w),
        }
        row.update(adj_fields)
        summary_rows.append(row)

        for k, v in sorted(verb_w.items(), key=lambda kv: -kv[1]):
            detail_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "weight_source": weight_source,
                    "label_type": "verb",
                    "label": k,
                    "weighted_score": v,
                    "share": v / verb_total,
                }
            )
        for k, v in sorted(adj_w.items(), key=lambda kv: -kv[1]):
            detail_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "weight_source": weight_source,
                    "label_type": "adjective",
                    "label": k,
                    "weighted_score": v,
                    "share": v / adj_total,
                }
            )

    summary_rows = sorted(
        summary_rows,
        key=lambda r: (
            str(r.get("task", "")),
            str(r.get("tokenizer", "")),
            -_to_float(r.get("bias_strength"), 0.0),
        ),
    )
    write_csv(out_dir / f"{task}_model_label_bias_from_attr_summary.csv", summary_rows)
    write_csv(out_dir / f"{task}_model_label_bias_from_attr_detail.csv", detail_rows)
    return summary_rows, detail_rows


def plot_model_x_label_bias_heatmaps(
    task: str,
    names: List[str],
    detail_rows: List[Dict[str, object]],
    out_dir: Path,
    top_k_labels_per_type: int = 20,
) -> None:
    if not HAS_MATPLOTLIB:
        print(f"[warn] matplotlib not found: skip model-x-label heatmaps ({task})")
        return

    rows = [r for r in detail_rows if str(r.get("task", "")) == task]
    if not rows:
        print(f"[warn] no rows for model-x-label heatmaps: task={task}")
        return

    models_seen = sorted({str(r.get("tokenizer", "")) for r in rows})
    models = [m for m in names if m in models_seen]
    if not models:
        models = models_seen
    if not models:
        print(f"[warn] no models for model-x-label heatmaps: task={task}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(18.0, max(5.5, 0.45 * len(models) + 2.2)), constrained_layout=True)
    label_types = ["verb", "adjective"]

    for ax, label_type in zip(axes, label_types):
        typed = [r for r in rows if str(r.get("label_type", "")) == label_type]
        if not typed:
            ax.set_axis_off()
            ax.set_title(f"{task} | {label_type} | no data")
            continue

        label_totals: Dict[str, float] = {}
        for r in typed:
            lb = str(r.get("label", ""))
            sh = _to_float(r.get("share"), 0.0)
            label_totals[lb] = label_totals.get(lb, 0.0) + sh
        labels_sorted = [k for k, _v in sorted(label_totals.items(), key=lambda kv: (-kv[1], kv[0]))]
        if int(top_k_labels_per_type) > 0:
            labels_sorted = labels_sorted[: int(top_k_labels_per_type)]
        if not labels_sorted:
            ax.set_axis_off()
            ax.set_title(f"{task} | {label_type} | no labels")
            continue

        share_map: Dict[Tuple[str, str], float] = {}
        for r in typed:
            m = str(r.get("tokenizer", ""))
            lb = str(r.get("label", ""))
            if lb in labels_sorted:
                share_map[(m, lb)] = _to_float(r.get("share"), 0.0)

        mat: List[List[float]] = []
        for m in models:
            mat.append([share_map.get((m, lb), 0.0) for lb in labels_sorted])

        im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
        ax.set_title(f"{task} | model x {label_type} labels (share)")
        ax.set_xticks(list(range(len(labels_sorted))))
        ax.set_xticklabels(labels_sorted, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(list(range(len(models))))
        ax.set_yticklabels(models)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("share")

    fig.savefig(out_dir / f"{task}_model_x_label_heatmaps.png", dpi=180)
    plt.close(fig)


def export_weighted_label_bias_summary(
    task: str,
    names: List[str],
    perturb_rows_by_model: Dict[str, List[Dict[str, str]]],
    class_rows_by_model: Dict[str, List[Dict[str, str]]],
    top_k_ids: int,
    out_dir: Path,
) -> None:
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    models = [n for n in names if n in perturb_rows_by_model and n in class_rows_by_model]
    for model in models:
        top_ids = _top_rows_for_model(perturb_rows_by_model[model], top_k_ids)
        class_rows = class_rows_by_model[model]
        by_id: Dict[int, List[Dict[str, str]]] = {}
        for r in class_rows:
            iid = int(r.get("item_id", 0))
            by_id.setdefault(iid, []).append(r)

        verb_w: Dict[str, float] = {}
        adj_w: Dict[str, float] = {}
        adj_by_verb: Dict[str, Dict[str, float]] = {}
        for rtop in top_ids:
            iid = int(rtop["item_id"])
            imp = max(0.0, _to_float(rtop["importance_score"], 0.0))
            for rr in by_id.get(iid, []):
                if _to_float(rr.get("samples_with_item_in_class"), 0.0) <= 0.0:
                    continue
                class_name = rr.get("class_name", "")
                verb, adj = _split_class_name(class_name)
                w = (
                    imp
                    * _to_float(rr.get("chi2_2x2"), 0.0)
                    # * _to_float(rr.get("sample_freq_given_class"), 0.0)
                )
                if w <= 0.0:
                    continue
                verb_w[verb] = verb_w.get(verb, 0.0) + w
                adj_w[adj] = adj_w.get(adj, 0.0) + w
                adj_by_verb.setdefault(verb, {})
                adj_by_verb[verb][adj] = adj_by_verb[verb].get(adj, 0.0) + w

        verb_total = sum(verb_w.values())
        adj_total = sum(adj_w.values())
        if verb_total > 0.0:
            top_verb = max(verb_w.items(), key=lambda kv: kv[1])[0]
            top_verb_share = verb_w[top_verb] / verb_total
            verb_concentration = 1.0 - _normalized_entropy(verb_w)
            summary_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "label_type": "verb",
                    "top_label": top_verb,
                    "top_share": top_verb_share,
                    "concentration": verb_concentration,
                    "bias_type": _bias_bucket(verb_concentration, top_verb_share),
                    "n_labels_used": len(verb_w),
                }
            )
            for k, v in sorted(verb_w.items(), key=lambda kv: -kv[1]):
                detail_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "label_type": "verb",
                        "label": k,
                        "weighted_score": v,
                        "share": v / verb_total,
                    }
                )
        if adj_total > 0.0:
            top_adj, top_adj_share, adj_concentration, n_adjs_used = _compute_adj_bias_within_verbs(adj_by_verb)
            summary_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "label_type": "adjective",
                    "top_label": top_adj,
                    "top_share": top_adj_share,
                    "concentration": adj_concentration,
                    "bias_type": _bias_bucket(adj_concentration, top_adj_share),
                    "n_labels_used": n_adjs_used,
                }
            )
            for k, v in sorted(adj_w.items(), key=lambda kv: -kv[1]):
                detail_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "label_type": "adjective",
                        "label": k,
                        "weighted_score": v,
                        "share": v / adj_total,
                    }
                )

    write_csv(out_dir / f"{task}_label_bias_weighted_summary.csv", summary_rows)
    write_csv(out_dir / f"{task}_label_bias_weighted_detail.csv", detail_rows)


def export_attr_weighted_label_bias_summary(
    task: str,
    names: List[str],
    attr_rows_by_model: Dict[str, List[Dict[str, str]]],
    class_rows_by_model: Dict[str, List[Dict[str, str]]],
    top_k_ids: int,
    out_dir: Path,
) -> None:
    summary_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []

    models = [n for n in names if n in attr_rows_by_model and n in class_rows_by_model]
    if not models:
        print(f"[warn] no attribution rows for attr-weighted label bias: task={task}")
        write_csv(out_dir / f"{task}_label_bias_attr_weighted_summary.csv", summary_rows)
        write_csv(out_dir / f"{task}_label_bias_attr_weighted_detail.csv", detail_rows)
        return

    for model in models:
        attr_rows = sorted(
            attr_rows_by_model[model],
            key=lambda r: (-abs(_to_float(r.get("signed_attr_sum"), 0.0)), int(r.get("item_id", 0))),
        )
        if top_k_ids > 0:
            attr_rows = attr_rows[:top_k_ids]
        attr_weight_by_id = {
            int(r["item_id"]): abs(_to_float(r.get("signed_attr_sum"), 0.0))
            for r in attr_rows
        }

        by_id: Dict[int, List[Dict[str, str]]] = {}
        for r in class_rows_by_model[model]:
            iid = int(r.get("item_id", 0))
            by_id.setdefault(iid, []).append(r)

        verb_w: Dict[str, float] = {}
        adj_w: Dict[str, float] = {}
        adj_by_verb: Dict[str, Dict[str, float]] = {}
        for iid, attr_w in attr_weight_by_id.items():
            if attr_w <= 0.0:
                continue
            for rr in by_id.get(iid, []):
                if _to_float(rr.get("samples_with_item_in_class"), 0.0) <= 0.0:
                    continue
                cls = rr.get("class_name", "")
                verb, adj = _split_class_name(cls)
                w = attr_w * _to_float(rr.get("chi2_2x2"), 0.0)
                if w <= 0.0:
                    continue
                verb_w[verb] = verb_w.get(verb, 0.0) + w
                adj_w[adj] = adj_w.get(adj, 0.0) + w
                adj_by_verb.setdefault(verb, {})
                adj_by_verb[verb][adj] = adj_by_verb[verb].get(adj, 0.0) + w

        verb_total = sum(verb_w.values())
        adj_total = sum(adj_w.values())
        if verb_total > 0.0:
            top_verb = max(verb_w.items(), key=lambda kv: kv[1])[0]
            top_verb_share = verb_w[top_verb] / verb_total
            verb_concentration = 1.0 - _normalized_entropy(verb_w)
            summary_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "weight_source": "id_attribution:signed_attr_sum(abs)",
                    "label_type": "verb",
                    "top_label": top_verb,
                    "top_share": top_verb_share,
                    "concentration": verb_concentration,
                    "bias_type": _bias_bucket(verb_concentration, top_verb_share),
                    "n_labels_used": len(verb_w),
                }
            )
            for k, v in sorted(verb_w.items(), key=lambda kv: -kv[1]):
                detail_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "weight_source": "id_attribution:signed_attr_sum(abs)",
                        "label_type": "verb",
                        "label": k,
                        "weighted_score": v,
                        "share": v / verb_total,
                    }
                )
        if adj_total > 0.0:
            top_adj, top_adj_share, adj_concentration, n_adjs_used = _compute_adj_bias_within_verbs(adj_by_verb)
            summary_rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "weight_source": "id_attribution:signed_attr_sum(abs)",
                    "label_type": "adjective",
                    "top_label": top_adj,
                    "top_share": top_adj_share,
                    "concentration": adj_concentration,
                    "bias_type": _bias_bucket(adj_concentration, top_adj_share),
                    "n_labels_used": n_adjs_used,
                }
            )
            for k, v in sorted(adj_w.items(), key=lambda kv: -kv[1]):
                detail_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "weight_source": "id_attribution:signed_attr_sum(abs)",
                        "label_type": "adjective",
                        "label": k,
                        "weighted_score": v,
                        "share": v / adj_total,
                    }
                )

    write_csv(out_dir / f"{task}_label_bias_attr_weighted_summary.csv", summary_rows)
    write_csv(out_dir / f"{task}_label_bias_attr_weighted_detail.csv", detail_rows)


def export_id_type_label_contribution_distribution(
    task: str,
    names: List[str],
    type_class_rows_by_model: Dict[str, List[Dict[str, str]]],
    out_dir: Path,
) -> None:
    verb_rows: List[Dict[str, object]] = []
    adj_rows: List[Dict[str, object]] = []
    verb_summary: List[Dict[str, object]] = []
    adj_summary: List[Dict[str, object]] = []

    models = [n for n in names if n in type_class_rows_by_model]
    for model in models:
        by_type_verb: Dict[Tuple[int, str], Dict[str, float]] = {}
        by_type_adj: Dict[Tuple[int, str], Dict[str, float]] = {}

        for r in type_class_rows_by_model[model]:
            if _to_float(r.get("samples_with_item_in_class"), 0.0) <= 0.0:
                continue
            t_id = int(_to_float(r.get("item_id"), 0))
            t_name = str(r.get("item_name", str(t_id)))
            cls = str(r.get("class_name", ""))
            verb, adj = _split_class_name(cls)
            # Use class occurrences so every id_type gets a meaningful distribution
            # even when chi2_2x2 becomes 0 for ubiquitous types.
            w = _to_float(r.get("occurrences_in_class"), 0.0)
            if w <= 0.0:
                continue

            kv = (t_id, t_name)
            if kv not in by_type_verb:
                by_type_verb[kv] = {}
            if kv not in by_type_adj:
                by_type_adj[kv] = {}
            by_type_verb[kv][verb] = by_type_verb[kv].get(verb, 0.0) + w
            by_type_adj[kv][adj] = by_type_adj[kv].get(adj, 0.0) + w

        for (t_id, t_name), m in by_type_verb.items():
            total = sum(m.values())
            if total <= 0.0:
                continue
            top_lb = max(m.items(), key=lambda kv: kv[1])[0]
            top_share = m[top_lb] / total
            conc = 1.0 - _normalized_entropy(m)
            verb_summary.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "id_type": t_id,
                    "id_type_name": t_name,
                    "top_verb": top_lb,
                    "top_share": top_share,
                    "concentration": conc,
                    "bias_type": _bias_bucket(conc, top_share),
                    "n_labels_used": len(m),
                }
            )
            for lb, w in sorted(m.items(), key=lambda kv: -kv[1]):
                verb_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "id_type": t_id,
                        "id_type_name": t_name,
                        "label_type": "verb",
                        "label": lb,
                        "weighted_score": w,
                        "share_within_id_type": w / total,
                    }
                )

        for (t_id, t_name), m in by_type_adj.items():
            total = sum(m.values())
            if total <= 0.0:
                continue
            top_lb = max(m.items(), key=lambda kv: kv[1])[0]
            top_share = m[top_lb] / total
            conc = 1.0 - _normalized_entropy(m)
            adj_summary.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "id_type": t_id,
                    "id_type_name": t_name,
                    "top_adjective": top_lb,
                    "top_share": top_share,
                    "concentration": conc,
                    "bias_type": _bias_bucket(conc, top_share),
                    "n_labels_used": len(m),
                }
            )
            for lb, w in sorted(m.items(), key=lambda kv: -kv[1]):
                adj_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "id_type": t_id,
                        "id_type_name": t_name,
                        "label_type": "adjective",
                        "label": lb,
                        "weighted_score": w,
                        "share_within_id_type": w / total,
                    }
                )

    write_csv(out_dir / f"{task}_id_type_verb_contribution_distribution.csv", verb_rows)
    write_csv(out_dir / f"{task}_id_type_adjective_contribution_distribution.csv", adj_rows)
    write_csv(out_dir / f"{task}_id_type_verb_contribution_summary.csv", verb_summary)
    write_csv(out_dir / f"{task}_id_type_adjective_contribution_summary.csv", adj_summary)


def export_id_label_contribution_distribution(
    task: str,
    names: List[str],
    class_rows_by_model: Dict[str, List[Dict[str, str]]],
    out_dir: Path,
) -> None:
    verb_rows: List[Dict[str, object]] = []
    adj_rows: List[Dict[str, object]] = []
    verb_summary: List[Dict[str, object]] = []
    adj_summary: List[Dict[str, object]] = []

    models = [n for n in names if n in class_rows_by_model]
    for model in models:
        by_id_verb: Dict[Tuple[int, str], Dict[str, float]] = {}
        by_id_adj: Dict[Tuple[int, str], Dict[str, float]] = {}
        for r in class_rows_by_model[model]:
            if _to_float(r.get("samples_with_item_in_class"), 0.0) <= 0.0:
                continue
            iid = int(_to_float(r.get("item_id"), 0))
            iname = str(r.get("item_name", str(iid)))
            cls = str(r.get("class_name", ""))
            verb, adj = _split_class_name(cls)
            # contribution proxy for this ID in this class
            w = _to_float(r.get("chi2_2x2"), 0.0) * _to_float(r.get("sample_freq_given_class"), 0.0)
            if w <= 0.0:
                continue
            key = (iid, iname)
            if key not in by_id_verb:
                by_id_verb[key] = {}
            if key not in by_id_adj:
                by_id_adj[key] = {}
            by_id_verb[key][verb] = by_id_verb[key].get(verb, 0.0) + w
            by_id_adj[key][adj] = by_id_adj[key].get(adj, 0.0) + w

        for (iid, iname), m in by_id_verb.items():
            total = sum(m.values())
            if total <= 0.0:
                continue
            top_lb = max(m.items(), key=lambda kv: kv[1])[0]
            top_share = m[top_lb] / total
            conc = 1.0 - _normalized_entropy(m)
            verb_summary.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "item_id": iid,
                    "item_name": iname,
                    "top_verb": top_lb,
                    "top_share": top_share,
                    "concentration": conc,
                    "bias_type": _bias_bucket(conc, top_share),
                    "n_labels_used": len(m),
                }
            )
            for lb, w in sorted(m.items(), key=lambda kv: -kv[1]):
                verb_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "item_id": iid,
                        "item_name": iname,
                        "label_type": "verb",
                        "label": lb,
                        "weighted_score": w,
                        "share_within_id": w / total,
                    }
                )

        for (iid, iname), m in by_id_adj.items():
            total = sum(m.values())
            if total <= 0.0:
                continue
            top_lb = max(m.items(), key=lambda kv: kv[1])[0]
            top_share = m[top_lb] / total
            conc = 1.0 - _normalized_entropy(m)
            adj_summary.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "item_id": iid,
                    "item_name": iname,
                    "top_adjective": top_lb,
                    "top_share": top_share,
                    "concentration": conc,
                    "bias_type": _bias_bucket(conc, top_share),
                    "n_labels_used": len(m),
                }
            )
            for lb, w in sorted(m.items(), key=lambda kv: -kv[1]):
                adj_rows.append(
                    {
                        "task": task,
                        "tokenizer": model,
                        "item_id": iid,
                        "item_name": iname,
                        "label_type": "adjective",
                        "label": lb,
                        "weighted_score": w,
                        "share_within_id": w / total,
                    }
                )

    write_csv(out_dir / f"{task}_id_verb_contribution_distribution.csv", verb_rows)
    write_csv(out_dir / f"{task}_id_adjective_contribution_distribution.csv", adj_rows)
    write_csv(out_dir / f"{task}_id_verb_contribution_summary.csv", verb_summary)
    write_csv(out_dir / f"{task}_id_adjective_contribution_summary.csv", adj_summary)

def plot_id_label_bias_map(
    task: str,
    bias_rows: List[Dict[str, object]],
    names: List[str],
    out_dir: Path,
) -> None:
    if not HAS_MATPLOTLIB:
        print(f"[warn] matplotlib not found: skip id-label bias map ({task})")
        return

    for model in names:
        rows = [r for r in bias_rows if r["task"] == task and r["tokenizer"] == model]
        rows = sorted(
            rows,
            key=lambda r: (-_to_float(r.get("bias_strength"), 0.0), int(r.get("item_id", 0))),
        )
        if not rows:
            continue
        xs = [_to_float(r["verb_concentration"], 0.0) for r in rows]
        ys = [_to_float(r["adj_concentration"], 0.0) for r in rows]

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.2), constrained_layout=True)
        ax.plot(xs, ys, ".", alpha=0.85)
        for r, x, y in zip(rows, xs, ys):
            ax.text(x, y, str(int(r["item_id"])), fontsize=7, alpha=0.8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("verb concentration (1=single verb, 0=broad)")
        ax.set_ylabel("adjective concentration (1=single adjective, 0=broad)")
        ax.set_title(f"{task} | {model} | ID label-bias map")
        ax.grid(alpha=0.25)
        fig.savefig(out_dir / f"{task}_{_sanitize_name(model)}_id_label_bias_map.png", dpi=180)
        plt.close(fig)


def _select_importance_rows_by_model(
    names: List[str],
    perturb_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_sample_rows_by_model: Dict[str, List[Dict[str, str]]],
    use_attr: bool,
    attr_sample_aggregation_mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
    top_k_ids: int,
) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    if use_attr:
        for model in names:
            sample_rows = attr_sample_rows_by_model.get(model, [])
            if sample_rows:
                total_by_id, _verb_by_id, _adj_by_id, _verb_adj_by_id = _aggregate_attr_samples_by_mode(
                    sample_rows=sample_rows,
                    mode=attr_sample_aggregation_mode,
                    sample_top_k_ids=sample_top_k_ids,
                    sample_top_p=sample_top_p,
                )
                pairs = sorted(total_by_id.items(), key=lambda kv: (-kv[1], kv[0]))
                if top_k_ids > 0:
                    pairs = pairs[:top_k_ids]
                out[model] = [
                    {
                        "item_id": int(iid),
                        "importance_score": float(total),
                    }
                    for iid, total in pairs
                ]
            else:
                rows = attr_rows_by_model.get(model, [])
                rows = sorted(
                    rows,
                    key=lambda r: (-abs(_to_float(r.get("signed_attr_sum"), 0.0)), int(r.get("item_id", 0))),
                )
                if top_k_ids > 0:
                    rows = rows[:top_k_ids]
                out[model] = [
                    {
                        "item_id": int(r["item_id"]),
                        "importance_score": abs(_to_float(r.get("signed_attr_sum"), 0.0)),
                    }
                    for r in rows
                ]
    else:
        for model in names:
            rows = perturb_rows_by_model.get(model, [])
            out[model] = _top_rows_for_model(rows, top_k_ids)
    return out


def _build_label_id_maps(
    importance_rows: List[Dict[str, object]],
    class_rows: List[Dict[str, str]],
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    selected_ids = {int(ir["item_id"]) for ir in importance_rows}
    by_id: Dict[int, List[Dict[str, str]]] = {}
    for r in class_rows:
        iid = int(r.get("item_id", 0))
        if selected_ids and iid not in selected_ids:
            continue
        by_id.setdefault(iid, []).append(r)

    verb_map: Dict[str, Dict[int, float]] = {}
    adj_map: Dict[str, Dict[int, float]] = {}
    for iid in sorted(by_id.keys()):
        for rr in by_id[iid]:
            if _to_float(rr.get("samples_with_item_in_class"), 0.0) <= 0.0:
                continue
            verb, adj = _split_class_name(rr.get("class_name", ""))
            # Align with *_id_*_contribution_distribution.csv weighting
            w = _to_float(rr.get("chi2_2x2"), 0.0) * _to_float(rr.get("sample_freq_given_class"), 0.0)
            if w <= 0.0:
                continue
            verb_map.setdefault(verb, {})
            verb_map[verb][iid] = verb_map[verb].get(iid, 0.0) + w
            adj_map.setdefault(adj, {})
            adj_map[adj][iid] = adj_map[adj].get(iid, 0.0) + w
    return verb_map, adj_map


def _build_label_id_maps_from_attr_samples(
    importance_rows: List[Dict[str, object]],
    attr_sample_rows: List[Dict[str, str]],
    attr_sample_aggregation_mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    selected_ids = {int(ir["item_id"]) for ir in importance_rows}
    _total_by_id, verb_by_id, adj_by_id, _verb_adj_by_id = _aggregate_attr_samples_by_mode(
        sample_rows=attr_sample_rows,
        mode=attr_sample_aggregation_mode,
        sample_top_k_ids=sample_top_k_ids,
        sample_top_p=sample_top_p,
    )
    verb_map: Dict[str, Dict[int, float]] = {}
    adj_map: Dict[str, Dict[int, float]] = {}
    for iid in selected_ids:
        for verb, w in verb_by_id.get(iid, {}).items():
            if w <= 0.0:
                continue
            verb_map.setdefault(verb, {})
            verb_map[verb][iid] = verb_map[verb].get(iid, 0.0) + w
        for adj, w in adj_by_id.get(iid, {}).items():
            if w <= 0.0:
                continue
            adj_map.setdefault(adj, {})
            adj_map[adj][iid] = adj_map[adj].get(iid, 0.0) + w
    return verb_map, adj_map


def _plot_single_label_id_heatmap(
    task: str,
    model: str,
    label_type: str,
    label_to_id_map: Dict[str, Dict[int, float]],
    top_k_ids: int,
    out_dir: Path,
    fixed_ids: List[int] | None = None,
) -> None:
    if not label_to_id_map:
        return

    label_scores = {lb: sum(idm.values()) for lb, idm in label_to_id_map.items()}
    labels = sorted(label_scores.keys(), key=lambda x: (-label_scores[x], x))
    if not labels:
        return

    id_scores: Dict[int, float] = {}
    for lb in labels:
        for iid, v in label_to_id_map[lb].items():
            id_scores[iid] = id_scores.get(iid, 0.0) + float(v)
    if fixed_ids:
        # Keep the bias-order IDs from id_label_bias_map even if a given ID has
        # zero weight within the selected top labels for this heatmap.
        ids = [int(i) for i in fixed_ids]
        if top_k_ids > 0:
            ids = ids[: max(1, top_k_ids)]
    else:
        ids = sorted(id_scores.keys(), key=lambda x: (-id_scores[x], x))[: max(1, top_k_ids)]
    if not ids:
        return

    id_totals: Dict[int, float] = {}
    for lb in labels:
        for iid in ids:
            id_totals[iid] = id_totals.get(iid, 0.0) + float(label_to_id_map[lb].get(iid, 0.0))

    mat: List[List[float]] = []
    for lb in labels:
        row: List[float] = []
        for iid in ids:
            w = float(label_to_id_map[lb].get(iid, 0.0))
            denom = id_totals.get(iid, 0.0)
            row.append((w / denom) if denom > 0.0 else 0.0)
        mat.append(row)

    rows: List[Dict[str, object]] = []
    for lb in labels:
        for iid in ids:
            w = float(label_to_id_map[lb].get(iid, 0.0))
            if w <= 0.0:
                continue
            rows.append(
                {
                    "task": task,
                    "tokenizer": model,
                    "label_type": label_type,
                    "label": lb,
                    "item_id": int(iid),
                    "weighted_score": w,
                    "share_within_id": (w / id_totals.get(iid, 1.0)) if id_totals.get(iid, 0.0) > 0.0 else 0.0,
                }
            )
    write_csv(out_dir / f"{task}_{_sanitize_name(model)}_{label_type}_id_bias_map.csv", rows)

    if not HAS_MATPLOTLIB:
        return
    fig_w = max(7.0, 0.52 * len(ids) + 3.0)
    fig_h = max(5.0, 0.35 * len(labels) + 2.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu")
    ax.set_title(f"{task} | {model} | {label_type}-ID bias map")
    ax.set_xlabel("ID")
    ax.set_ylabel(label_type)
    ax.set_xticks(list(range(len(ids))))
    ax.set_xticklabels([str(i) for i in ids], rotation=80)
    ax.set_yticks(list(range(len(labels))))
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("share_within_id")
    fig.savefig(out_dir / f"{task}_{_sanitize_name(model)}_{label_type}_id_bias_map.png", dpi=180)
    plt.close(fig)


def plot_label_id_bias_maps_per_model(
    task: str,
    names: List[str],
    perturb_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_rows_by_model: Dict[str, List[Dict[str, str]]],
    class_rows_by_model: Dict[str, List[Dict[str, str]]],
    attr_sample_rows_by_model: Dict[str, List[Dict[str, str]]],
    use_attr: bool,
    attr_sample_aggregation_mode: str,
    sample_top_k_ids: int,
    sample_top_p: float,
    top_k_ids: int,
    out_dir: Path,
    bias_rows: List[Dict[str, object]] | None = None,
) -> None:
    importance_by_model = _select_importance_rows_by_model(
        names=names,
        perturb_rows_by_model=perturb_rows_by_model,
        attr_rows_by_model=attr_rows_by_model,
        attr_sample_rows_by_model=attr_sample_rows_by_model,
        use_attr=use_attr,
        attr_sample_aggregation_mode=attr_sample_aggregation_mode,
        sample_top_k_ids=sample_top_k_ids,
        sample_top_p=sample_top_p,
        top_k_ids=top_k_ids,
    )
    for model in names:
        class_rows = class_rows_by_model.get(model, [])
        attr_sample_rows = attr_sample_rows_by_model.get(model, [])
        imp_rows = importance_by_model.get(model, [])
        if not imp_rows:
            continue
        fixed_ids_verb: List[int] = [int(r["item_id"]) for r in imp_rows]
        fixed_ids_adj: List[int] = [int(r["item_id"]) for r in imp_rows]
        if use_attr and attr_sample_rows:
            verb_map, adj_map = _build_label_id_maps_from_attr_samples(
                imp_rows,
                attr_sample_rows,
                attr_sample_aggregation_mode,
                sample_top_k_ids,
                sample_top_p,
            )
        else:
            if not class_rows:
                continue
            verb_map, adj_map = _build_label_id_maps(imp_rows, class_rows)
        _plot_single_label_id_heatmap(
            task=task,
            model=model,
            label_type="verb",
            label_to_id_map=verb_map,
            top_k_ids=top_k_ids,
            out_dir=out_dir,
            fixed_ids=fixed_ids_verb,
        )
        _plot_single_label_id_heatmap(
            task=task,
            model=model,
            label_type="adjective",
            label_to_id_map=adj_map,
            top_k_ids=top_k_ids,
            out_dir=out_dir,
            fixed_ids=fixed_ids_adj,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", type=str, default="experiments/bandai")
    ap.add_argument("--out_suffix", type=str, default="id_contrib_test")
    ap.add_argument("--attr_suffix", type=str, default="")
    ap.add_argument("--out_dir", type=str, default="experiments/bandai/id_contrib_viz")
    ap.add_argument("--names", type=str, nargs="+", default=DEFAULT_NAMES)
    ap.add_argument("--top_k_ids", type=int, default=20)
    ap.add_argument("--top_k_ids_per_class", type=int, default=5)
    ap.add_argument("--top_k_classes", type=int, default=20)
    ap.add_argument(
        "--attr_sample_aggregation_mode",
        type=str,
        default="abs_sum",
        choices=["abs_sum", "sample_topk_count", "sample_top_p_count"],
        help="How to aggregate id_attribution_per_sample for bias/ID selection.",
    )
    ap.add_argument(
        "--sample_top_k_ids",
        type=int,
        default=5,
        help="Used when --attr_sample_aggregation_mode=sample_topk_count.",
    )
    ap.add_argument(
        "--sample_top_p",
        type=float,
        default=0.0,
        help="If 0 or >=1, disabled. If 0<p<1, apply per-sample cumulative top-p filtering.",
    )
    ap.add_argument("--bias_strength_bin_width", type=float, default=0.1)
    ap.add_argument("--motif_min_support", type=float, default=0.35)
    ap.add_argument("--motif_min_samples", type=int, default=3)
    ap.add_argument("--motif_top_k_ids", type=int, default=8)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, perturb_rows, class_rows, type_class_rows, attr_rows, attr_sample_rows = load_all_data(
        base_dir,
        args.out_suffix,
        args.attr_suffix,
        args.names,
    )
    write_base_metrics_csv(summaries, args.names, out_dir / "base_metrics_comparison.csv")

    plot_base_metrics(summaries, args.names, out_dir / "base_metrics_comparison.png")
    action_top_rows = plot_top_id_importance_per_model(
        "actionrec",
        perturb_rows["actionrec"],
        args.names,
        args.top_k_ids,
        out_dir,
    )
    retrieval_top_rows = plot_top_id_importance_per_model(
        "retrieval",
        perturb_rows["retrieval"],
        args.names,
        args.top_k_ids,
        out_dir,
    )
    write_csv(out_dir / "actionrec_top_id_importance_per_model.csv", action_top_rows)
    write_csv(out_dir / "retrieval_top_id_importance_per_model.csv", retrieval_top_rows)

    action_assoc_rows = export_class_id_association_per_model(
        "actionrec",
        class_rows["actionrec"],
        args.names,
        args.top_k_ids_per_class,
        out_dir,
    )
    retrieval_assoc_rows = export_class_id_association_per_model(
        "retrieval",
        class_rows["retrieval"],
        args.names,
        args.top_k_ids_per_class,
        out_dir,
    )
    plot_class_id_association_heatmap(
        "actionrec",
        action_assoc_rows,
        args.names,
        args.top_k_classes,
        args.top_k_ids_per_class,
        out_dir,
    )
    plot_class_id_association_heatmap(
        "retrieval",
        retrieval_assoc_rows,
        args.names,
        args.top_k_classes,
        args.top_k_ids_per_class,
        out_dir,
    )

    has_attr_action = bool(attr_rows["actionrec"]) and bool(attr_sample_rows["actionrec"])
    has_attr_retrieval = bool(attr_rows["retrieval"]) and bool(attr_sample_rows["retrieval"])
    if has_attr_action:
        action_bias_rows = export_id_label_bias_from_attr_per_model(
            "actionrec",
            args.names,
            attr_rows["actionrec"],
            attr_sample_rows["actionrec"],
            args.top_k_ids,
            args.attr_sample_aggregation_mode,
            args.sample_top_k_ids,
            args.sample_top_p,
            out_dir,
        )
    else:
        action_bias_rows = export_id_label_bias_per_model(
            "actionrec",
            args.names,
            perturb_rows["actionrec"],
            class_rows["actionrec"],
            args.top_k_ids,
            out_dir,
        )
    if has_attr_retrieval:
        retrieval_bias_rows = export_id_label_bias_from_attr_per_model(
            "retrieval",
            args.names,
            attr_rows["retrieval"],
            attr_sample_rows["retrieval"],
            args.top_k_ids,
            args.attr_sample_aggregation_mode,
            args.sample_top_k_ids,
            args.sample_top_p,
            out_dir,
        )
    else:
        retrieval_bias_rows = export_id_label_bias_per_model(
            "retrieval",
            args.names,
            perturb_rows["retrieval"],
            class_rows["retrieval"],
            args.top_k_ids,
            out_dir,
        )
    plot_id_label_bias_map("actionrec", action_bias_rows, args.names, out_dir)
    plot_id_label_bias_map("retrieval", retrieval_bias_rows, args.names, out_dir)
    export_id_type_label_contribution_distribution(
        "actionrec",
        args.names,
        type_class_rows["actionrec"],
        out_dir,
    )
    export_id_type_label_contribution_distribution(
        "retrieval",
        args.names,
        type_class_rows["retrieval"],
        out_dir,
    )
    export_id_label_contribution_distribution(
        "actionrec",
        args.names,
        class_rows["actionrec"],
        out_dir,
    )
    export_id_label_contribution_distribution(
        "retrieval",
        args.names,
        class_rows["retrieval"],
        out_dir,
    )
    export_bias_strength_distribution_and_comparison(
        "actionrec",
        args.names,
        action_bias_rows,
        out_dir,
        args.bias_strength_bin_width,
    )
    export_bias_strength_distribution_and_comparison(
        "retrieval",
        args.names,
        retrieval_bias_rows,
        out_dir,
        args.bias_strength_bin_width,
    )
    export_role_disentanglement_comparison("actionrec", args.names, action_bias_rows, out_dir)
    export_role_disentanglement_comparison("retrieval", args.names, retrieval_bias_rows, out_dir)
    export_conditional_id_set_motifs(
        "actionrec",
        args.names,
        attr_sample_rows["actionrec"],
        args.attr_sample_aggregation_mode,
        args.sample_top_k_ids,
        args.sample_top_p,
        args.motif_min_support,
        args.motif_min_samples,
        args.motif_top_k_ids,
        out_dir,
    )
    export_conditional_id_set_motifs(
        "retrieval",
        args.names,
        attr_sample_rows["retrieval"],
        args.attr_sample_aggregation_mode,
        args.sample_top_k_ids,
        args.sample_top_p,
        args.motif_min_support,
        args.motif_min_samples,
        args.motif_top_k_ids,
        out_dir,
    )
    plot_conditional_id_set_support_bars("actionrec", args.names, out_dir)
    plot_conditional_id_set_support_bars("retrieval", args.names, out_dir)
    use_attr = has_attr_action
    plot_label_id_bias_maps_per_model(
        "actionrec",
        args.names,
        perturb_rows["actionrec"],
        attr_rows["actionrec"],
        class_rows["actionrec"],
        attr_sample_rows["actionrec"],
        use_attr,
        args.attr_sample_aggregation_mode,
        args.sample_top_k_ids,
        args.sample_top_p,
        args.top_k_ids,
        out_dir,
        action_bias_rows,
    )
    use_attr = has_attr_retrieval
    plot_label_id_bias_maps_per_model(
        "retrieval",
        args.names,
        perturb_rows["retrieval"],
        attr_rows["retrieval"],
        class_rows["retrieval"],
        attr_sample_rows["retrieval"],
        use_attr,
        args.attr_sample_aggregation_mode,
        args.sample_top_k_ids,
        args.sample_top_p,
        args.top_k_ids,
        out_dir,
        retrieval_bias_rows,
    )
    export_weighted_label_bias_summary(
        "actionrec",
        args.names,
        perturb_rows["actionrec"],
        class_rows["actionrec"],
        args.top_k_ids,
        out_dir,
    )
    export_weighted_label_bias_summary(
        "retrieval",
        args.names,
        perturb_rows["retrieval"],
        class_rows["retrieval"],
        args.top_k_ids,
        out_dir,
    )
    if has_attr_action and has_attr_retrieval:
        export_attr_weighted_label_bias_summary(
            "actionrec",
            args.names,
            attr_rows["actionrec"],
            class_rows["actionrec"],
            args.top_k_ids,
            out_dir,
        )
        export_attr_weighted_label_bias_summary(
            "retrieval",
            args.names,
            attr_rows["retrieval"],
            class_rows["retrieval"],
            args.top_k_ids,
            out_dir,
        )
    if has_attr_action:
        _action_model_bias_summary, action_model_bias_detail = export_model_label_bias_from_attr_per_model(
            "actionrec",
            args.names,
            attr_rows["actionrec"],
            attr_sample_rows["actionrec"],
            args.top_k_ids,
            args.attr_sample_aggregation_mode,
            args.sample_top_k_ids,
            args.sample_top_p,
            out_dir,
        )
        plot_model_x_label_bias_heatmaps(
            "actionrec",
            args.names,
            action_model_bias_detail,
            out_dir,
        )
    if has_attr_retrieval:
        _retrieval_model_bias_summary, retrieval_model_bias_detail = export_model_label_bias_from_attr_per_model(
            "retrieval",
            args.names,
            attr_rows["retrieval"],
            attr_sample_rows["retrieval"],
            args.top_k_ids,
            args.attr_sample_aggregation_mode,
            args.sample_top_k_ids,
            args.sample_top_p,
            out_dir,
        )
        plot_model_x_label_bias_heatmaps(
            "retrieval",
            args.names,
            retrieval_model_bias_detail,
            out_dir,
        )

    if not HAS_MATPLOTLIB:
        print("[info] PNG output skipped because matplotlib is not installed.")
        print("[info] install with: pip install matplotlib")
    print(f"[done] wrote visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
