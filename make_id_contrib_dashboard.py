#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _merge_task_summary(viz_dir: Path, task: str) -> pd.DataFrame:
    base = _safe_read_csv(viz_dir / 'base_metrics_comparison.csv')
    if base.empty:
        raise FileNotFoundError(f'missing {viz_dir / "base_metrics_comparison.csv"}')
    base = base[base['task'] == task].copy()
    if base.empty:
        raise ValueError(f'no rows for task={task} in base_metrics_comparison.csv')

    bias = _safe_read_csv(viz_dir / f'{task}_bias_strength_comparison.csv')
    top_imp = _safe_read_csv(viz_dir / f'{task}_top_id_importance_per_model.csv')
    class_assoc = _safe_read_csv(viz_dir / f'{task}_class_id_association_per_model.csv')
    attr_summary = _safe_read_csv(viz_dir / f'{task}_model_label_bias_from_attr_summary.csv')
    role_comp = _safe_read_csv(viz_dir / f'{task}_role_disentanglement_comparison.csv')

    if task == 'actionrec':
        base['primary_metric'] = base['acc']
        base['secondary_metric'] = base['macro_f1']
        base['primary_name'] = 'acc'
        base['secondary_name'] = 'macro_f1'
    else:
        base['primary_metric'] = base['mean_R@1']
        base['secondary_metric'] = 0.5 * (base.get('t2m_R@1', 0.0) + base.get('m2t_R@1', 0.0))
        base['primary_name'] = 'mean_R@1'
        base['secondary_name'] = 'mean(R@1)'

    if not top_imp.empty:
        g = top_imp.groupby('tokenizer', as_index=False).agg(
            n_top_ids=('item_id', 'count'),
            top1_importance=('importance_score', 'max'),
            top5_importance_sum=('importance_score', lambda s: float(s.nlargest(min(5, len(s))).sum())),
            top10_importance_sum=('importance_score', lambda s: float(s.nlargest(min(10, len(s))).sum())),
            mean_top_importance=('importance_score', 'mean'),
        )
        base = base.merge(g, on='tokenizer', how='left')

    if not class_assoc.empty:
        g = class_assoc.groupby('tokenizer', as_index=False).agg(
            n_assoc_rows=('item_id', 'count'),
            mean_assoc_chi2=('chi2_2x2', 'mean'),
            max_assoc_chi2=('chi2_2x2', 'max'),
            mean_assoc_pmi=('pmi_bits', 'mean'),
            n_classes_with_assoc=('class_name', 'nunique'),
        )
        rank1 = class_assoc[class_assoc['rank'] == 1].groupby('tokenizer', as_index=False).agg(
            mean_rank1_chi2=('chi2_2x2', 'mean'),
            mean_rank1_pmi=('pmi_bits', 'mean'),
        )
        g = g.merge(rank1, on='tokenizer', how='left')
        base = base.merge(g, on='tokenizer', how='left')

    if not bias.empty:
        base = base.merge(bias, on=['task', 'tokenizer'], how='left')
    if not attr_summary.empty and 'tokenizer' in attr_summary.columns:
        keep_cols = [c for c in ['tokenizer', 'verb_concentration', 'adj_concentration', 'adj_global_concentration', 'verb_within_adj_concentration', 'adj_conditional_gap', 'verb_conditional_gap', 'role_dominance_sym', 'role_abs_dominance_sym'] if c in attr_summary.columns]
        if keep_cols:
            base = base.merge(attr_summary[keep_cols], on='tokenizer', how='left')
    if not role_comp.empty and 'tokenizer' in role_comp.columns:
        base = base.merge(role_comp, on=['task', 'tokenizer'], how='left') if 'task' in role_comp.columns else base.merge(role_comp, on='tokenizer', how='left')

    # convenience ranks
    base = base.sort_values(['primary_metric', 'secondary_metric'], ascending=[False, False]).reset_index(drop=True)
    base['perf_rank'] = range(1, len(base) + 1)
    if 'mean_bias_strength' in base.columns:
        base['low_bias_rank'] = base['mean_bias_strength'].rank(method='min', ascending=True)

    return base


def _text_table(
    ax,
    df: pd.DataFrame,
    title: str,
    cols: List[str],
    max_rows: int = 12,
    col_labels: Optional[Dict[str, str]] = None,
) -> None:
    ax.axis('off')
    show = df[cols].copy().head(max_rows)
    for c in show.columns:
        if pd.api.types.is_numeric_dtype(show[c]):
            show[c] = show[c].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
    labels = [col_labels.get(c, c) if col_labels else c for c in show.columns]
    table = ax.table(
        cellText=show.values,
        colLabels=labels,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    ax.set_title(title)


def _bar(ax, labels: List[str], values: List[float], title: str, ylabel: str) -> None:
    x = range(len(labels))
    ax.bar(x, values)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)


def _grouped_bars(ax, labels: List[str], series: List[tuple], title: str, ylabel: str) -> None:
    if not series:
        ax.axis('off')
        return
    n = len(series)
    width = 0.8 / max(1, n)
    x = list(range(len(labels)))
    for i, (name, vals) in enumerate(series):
        xs = [xx - 0.4 + width / 2.0 + i * width for xx in x]
        ax.bar(xs, vals, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)
    ax.legend(fontsize=8)


def _scatter(ax, x, y, labels, title, xlabel, ylabel):
    ax.scatter(x, y)
    for xi, yi, lb in zip(x, y, labels):
        if pd.isna(xi) or pd.isna(yi):
            continue
        ax.annotate(str(lb), (xi, yi), fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def _make_dashboard_for_task(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    labels = df['tokenizer'].tolist()
    primary_name = str(df['primary_name'].iloc[0])
    secondary_name = str(df['secondary_name'].iloc[0])

    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    _bar(ax1, labels, df['primary_metric'].tolist(), f'{task}: primary performance', primary_name)

    ax2 = fig.add_subplot(gs[0, 1])
    _bar(ax2, labels, df['secondary_metric'].tolist(), f'{task}: secondary performance', secondary_name)

    ax3 = fig.add_subplot(gs[0, 2])
    if 'mean_bias_strength' in df.columns:
        _bar(ax3, labels, df['mean_bias_strength'].fillna(float('nan')).tolist(), f'{task}: mean bias strength', 'mean_bias_strength')
    else:
        ax3.axis('off')
        ax3.set_title('bias summary missing')

    ax4 = fig.add_subplot(gs[1, 0])
    if 'top5_importance_sum' in df.columns:
        _bar(ax4, labels, df['top5_importance_sum'].fillna(float('nan')).tolist(), f'{task}: top-5 importance sum', 'sum importance')
    else:
        ax4.axis('off')
        ax4.set_title('top importance missing')

    ax5 = fig.add_subplot(gs[1, 1])
    if 'mean_rank1_chi2' in df.columns:
        _bar(ax5, labels, df['mean_rank1_chi2'].fillna(float('nan')).tolist(), f'{task}: mean rank-1 chi2', 'chi2')
    else:
        ax5.axis('off')
        ax5.set_title('class association missing')

    ax6 = fig.add_subplot(gs[1, 2])
    if 'mean_bias_strength' in df.columns:
        _scatter(
            ax6,
            df['mean_bias_strength'],
            df['primary_metric'],
            df['tokenizer'],
            f'{task}: performance vs bias',
            'mean_bias_strength (low is less concentrated)',
            primary_name,
        )
    else:
        ax6.axis('off')
        ax6.set_title('bias summary missing')

    ax7 = fig.add_subplot(gs[2, :])
    cols = ['tokenizer', 'perf_rank', 'primary_metric', 'secondary_metric']
    for extra in ['mean_bias_strength', 'mean_verb_concentration', 'mean_adj_concentration', 'top1_importance', 'top5_importance_sum', 'mean_rank1_chi2']:
        if extra in df.columns:
            cols.append(extra)
    _text_table(ax7, df, f'{task}: one-look summary table', cols)

    fig.suptitle(f'ID contribution dashboard: {task}', fontsize=16)
    fig.savefig(out_dir / f'{task}_look_at_once_dashboard.png', dpi=180)
    plt.close(fig)


def _make_role_disentanglement_dashboard(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    labels = df['tokenizer'].tolist()
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    series = []
    for name, col in [('verb(global)', 'verb_concentration'), ('adj(global)', 'adj_global_concentration'), ('adj|verb', 'adj_concentration'), ('verb|adj', 'verb_within_adj_concentration')]:
        if col in df.columns:
            series.append((name, df[col].tolist()))
    _grouped_bars(ax1, labels, series, f'{task}: role concentrations', 'concentration') if series else ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    if 'weighted_mean_abs_role_dominance_sym' in df.columns:
        _bar(ax2, labels, df['weighted_mean_abs_role_dominance_sym'].tolist(), f'{task}: mean |role dominance|', '|dominance|')
    else:
        ax2.axis('off')
        ax2.set_title('role dominance missing')

    ax3 = fig.add_subplot(gs[0, 2])
    series = []
    for name, col in [('adj gap', 'weighted_mean_adj_conditional_gap'), ('verb gap', 'weighted_mean_verb_conditional_gap')]:
        if col in df.columns:
            series.append((name, df[col].tolist()))
    _grouped_bars(ax3, labels, series, f'{task}: conditional gaps', 'gap') if series else ax3.axis('off')

    ax4 = fig.add_subplot(gs[1, 0])
    series = []
    for name, col in [('verb_only', 'weighted_verb_only_ratio'), ('adj_only', 'weighted_adj_only_ratio'), ('mixed', 'weighted_mixed_ratio'), ('generic', 'weighted_generic_ratio')]:
        if col in df.columns:
            series.append((name, df[col].tolist()))
    if series:
        _grouped_bars(ax4, labels, series, f'{task}: role type ratios', 'ratio (log scale)')
        pos_vals = [float(v) for _, vals in series for v in vals if pd.notna(v) and float(v) > 0.0]
        if pos_vals:
            ax4.set_yscale('log')
            ax4.set_ylim(bottom=max(1e-4, min(pos_vals) * 0.5))
        else:
            ax4.set_title(f'{task}: role type ratios (no positive values)')
    else:
        ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1:])
    if 'weighted_mean_abs_role_dominance_sym' in df.columns:
        _scatter(ax5, df['weighted_mean_abs_role_dominance_sym'], df['primary_metric'], df['tokenizer'], f'{task}: perf vs |role dominance|', '|role dominance|', str(df['primary_name'].iloc[0]))
    else:
        ax5.axis('off')
        ax5.set_title('role dominance missing')

    ax6 = fig.add_subplot(gs[2, :])
    cols = ['tokenizer', 'primary_metric', 'weighted_mean_abs_role_dominance_sym', 'weighted_mean_adj_conditional_gap', 'weighted_mean_verb_conditional_gap', 'weighted_verb_only_ratio', 'weighted_adj_only_ratio', 'weighted_mixed_ratio', 'weighted_generic_ratio']
    cols = [c for c in cols if c in df.columns]
    col_labels = {
        'weighted_mean_abs_role_dominance_sym': 'weighted_mean_abs\n_role_dominance_sym',
        'weighted_mean_adj_conditional_gap': 'weighted_mean_adj\n_conditional_gap',
        'weighted_mean_verb_conditional_gap': 'weighted_mean_verb\n_conditional_gap',
        'weighted_verb_only_ratio': 'weighted_verb_only\n_ratio',
        'weighted_adj_only_ratio': 'weighted_adj_only\n_ratio',
        'weighted_mixed_ratio': 'weighted_mixed\n_ratio',
        'weighted_generic_ratio': 'weighted_generic\n_ratio',
    }
    _text_table(ax6, df, f'{task}: role disentanglement summary', cols, col_labels=col_labels)

    fig.suptitle(f'Role disentanglement dashboard: {task}', fontsize=16)
    fig.savefig(out_dir / f'{task}_role_disentanglement_dashboard.png', dpi=180)
    plt.close(fig)


def _make_cross_task_summary(action_df: pd.DataFrame, retrieval_df: pd.DataFrame, out_dir: Path) -> None:
    a = action_df[['tokenizer', 'primary_metric', 'secondary_metric']].rename(columns={
        'primary_metric': 'actionrec_acc',
        'secondary_metric': 'actionrec_macro_f1',
    })
    r = retrieval_df[['tokenizer', 'primary_metric', 'secondary_metric']].rename(columns={
        'primary_metric': 'retrieval_mean_R1',
        'secondary_metric': 'retrieval_mean_R1_dup',
    })
    out = a.merge(r, on='tokenizer', how='outer')
    if 'mean_bias_strength' in action_df.columns:
        out = out.merge(action_df[['tokenizer', 'mean_bias_strength']].rename(columns={'mean_bias_strength': 'actionrec_mean_bias'}), on='tokenizer', how='left')
    if 'mean_bias_strength' in retrieval_df.columns:
        out = out.merge(retrieval_df[['tokenizer', 'mean_bias_strength']].rename(columns={'mean_bias_strength': 'retrieval_mean_bias'}), on='tokenizer', how='left')
    if 'top5_importance_sum' in action_df.columns:
        out = out.merge(action_df[['tokenizer', 'top5_importance_sum']].rename(columns={'top5_importance_sum': 'actionrec_top5_importance_sum'}), on='tokenizer', how='left')
    if 'top5_importance_sum' in retrieval_df.columns:
        out = out.merge(retrieval_df[['tokenizer', 'top5_importance_sum']].rename(columns={'top5_importance_sum': 'retrieval_top5_importance_sum'}), on='tokenizer', how='left')
    if 'mean_rank1_chi2' in action_df.columns:
        out = out.merge(action_df[['tokenizer', 'mean_rank1_chi2']].rename(columns={'mean_rank1_chi2': 'actionrec_mean_rank1_chi2'}), on='tokenizer', how='left')
    if 'mean_rank1_chi2' in retrieval_df.columns:
        out = out.merge(retrieval_df[['tokenizer', 'mean_rank1_chi2']].rename(columns={'mean_rank1_chi2': 'retrieval_mean_rank1_chi2'}), on='tokenizer', how='left')
    out.to_csv(out_dir / 'look_at_once_summary_table.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    if 'actionrec_mean_bias' in out.columns:
        _scatter(axes[0], out['actionrec_mean_bias'], out['actionrec_acc'], out['tokenizer'], 'ActionRec: acc vs bias', 'mean_bias_strength', 'acc')
    else:
        axes[0].axis('off')
    if 'retrieval_mean_bias' in out.columns:
        _scatter(axes[1], out['retrieval_mean_bias'], out['retrieval_mean_R1'], out['tokenizer'], 'Retrieval: mean_R@1 vs bias', 'mean_bias_strength', 'mean_R@1')
    else:
        axes[1].axis('off')
    fig.savefig(out_dir / 'look_at_once_cross_task_scatter.png', dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--viz_dir', type=str, default='experiments/bandai/id_contrib_viz', help='directory containing outputs from visualize_id_contrib_bandai2.py')
    ap.add_argument('--out_dir', type=str, default='', help='output directory; default = viz_dir/look_at_once')
    args = ap.parse_args()

    viz_dir = Path(args.viz_dir)
    out_dir = Path(args.out_dir) if str(args.out_dir).strip() else (viz_dir / 'look_at_once')
    out_dir.mkdir(parents=True, exist_ok=True)

    action_df = _merge_task_summary(viz_dir, 'actionrec')
    retrieval_df = _merge_task_summary(viz_dir, 'retrieval')

    action_df.to_csv(out_dir / 'actionrec_look_at_once_summary.csv', index=False)
    retrieval_df.to_csv(out_dir / 'retrieval_look_at_once_summary.csv', index=False)

    _make_dashboard_for_task(action_df, 'actionrec', out_dir)
    _make_dashboard_for_task(retrieval_df, 'retrieval', out_dir)
    _make_role_disentanglement_dashboard(action_df, 'actionrec', out_dir)
    _make_role_disentanglement_dashboard(retrieval_df, 'retrieval', out_dir)
    _make_cross_task_summary(action_df, retrieval_df, out_dir)

    print(f'[ok] wrote dashboards to {out_dir}')


if __name__ == '__main__':
    main()
