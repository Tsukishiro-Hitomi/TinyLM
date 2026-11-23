from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = Path('runs/batch_size_comparison')
out.mkdir(parents=True, exist_ok=True)
for field, label, output in [('tokens', 'Training tokens', 'tokens'), ('wall_time_s', 'Wall time (s)', 'time')]:
    fig, ax = plt.subplots()
    for path in sorted(Path('runs').glob('batch_*_seed336/metrics.csv')):
        batch = int(path.parent.name.split('_')[1])
        with path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        xs = [int(r['step']) * batch * 256 if field == 'tokens' else float(r[field]) for r in rows]
        ax.plot(xs, [float(r['val_loss']) for r in rows], label=str(batch))
    ax.set(xlabel=label, ylabel='Validation loss', title='TinyStories batch-size sweep')
    ax.legend(title='Batch size')
    fig.tight_layout()
    fig.savefig(out / f'validation_vs_{output}.png', dpi=160)
    plt.close(fig)
