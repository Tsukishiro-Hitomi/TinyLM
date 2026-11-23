from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('runs/learning_rate_comparison')
out.mkdir(parents=True, exist_ok=True)
for xcol, xlabel, name in [('step', 'Training steps', 'steps'), ('wall_time_s', 'Wall time (seconds)', 'time')]:
    fig, ax = plt.subplots()
    for path in sorted(Path('runs').glob('lr_sweep_*/metrics.csv')):
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            ax.plot([float(r[xcol]) for r in rows], [float(r['val_loss']) for r in rows], label=path.parent.name.removeprefix('lr_sweep_').removesuffix('_seed336'))
    ax.set(xlabel=xlabel, ylabel='Validation loss', title='TinyStories learning-rate sweep')
    ax.legend(title='Peak LR')
    fig.tight_layout()
    fig.savefig(out / f'validation_vs_{name}.png', dpi=160)
    plt.close(fig)
