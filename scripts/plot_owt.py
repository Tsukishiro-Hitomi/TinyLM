import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('runs/owt_comparison')
out.mkdir(parents=True, exist_ok=True)
for x, label in [('step', 'Training steps'), ('wall_time_s', 'Wall time (seconds)')]:
    fig, ax = plt.subplots()
    for run in ('owt_baseline_seed336', 'owt_tied_seed336'):
        path = Path('runs') / run / 'metrics.csv'
        if not path.exists():
            continue
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            ax.plot([float(r[x]) for r in rows], [float(r['val_loss']) for r in rows], label=run)
    ax.set(xlabel=label, ylabel='Validation loss', title='OpenWebText')
    ax.legend(fontsize='small')
    fig.tight_layout()
    fig.savefig(out / f'validation_vs_{x}.png', dpi=160)
    plt.close(fig)
