from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('runs/ablation_comparison')
out.mkdir(parents=True, exist_ok=True)
baseline = Path('runs/lr_sweep_0.001_seed336/metrics.csv')
for name, paths in {
    'normalization': [baseline, Path('runs/ablation_no_norm_seed336/metrics.csv'), Path('runs/ablation_no_norm_low_lr_seed336/metrics.csv'), Path('runs/ablation_post_norm_seed336/metrics.csv')],
    'position': [baseline, Path('runs/ablation_no_rope_seed336/metrics.csv')],
    'activation': [baseline, Path('runs/ablation_silu_seed336/metrics.csv')],
}.items():
    fig, ax = plt.subplots()
    for path in paths:
        if not path.exists():
            continue
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            ax.plot([int(r['step']) for r in rows], [float(r['val_loss']) for r in rows], label=path.parent.name)
    ax.set(xlabel='Training steps', ylabel='Validation loss', title=name)
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(out / f'{name}.png', dpi=160)
    plt.close(fig)
