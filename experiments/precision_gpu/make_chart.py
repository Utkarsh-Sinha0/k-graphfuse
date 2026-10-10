"""Render only the six audited T4 comparisons; no new benchmark runs."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

base = Path(__file__).parent
report = json.loads((base / 'PRODUCT_VERIFIED.json').read_text())
rows = report['t4_cases']
assert len(rows) == 6 and all(r['dual_gate'] for r in rows)
labels = [f"{r['n']:,} vertices · {r['q']} channels\n" +
          ('Value + derivative' if r['sensitivity'] else 'Polynomial filter') for r in rows]
speed = np.array([r['speedup'] for r in rows])
error = np.array([r['worst_component_error_reduction_percent'] for r in rows])
assert np.all(speed > 1) and np.all(error > 0)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.spines.left': False})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 8), gridspec_kw={'width_ratios': [1.4, 1]})
y = np.arange(len(rows))
ax.barh(y, speed, color='#147DAD', height=.55)
# Block-median intervals need not contain the ratio of pooled medians.
lower = np.array([r['ci_low'] for r in rows]); upper = np.array([r['ci_high'] for r in rows])
assert np.all(lower <= upper)
ax.hlines(y, lower, upper, color='#17334D', linewidth=1.5)
ax.vlines(lower, y-.08, y+.08, color='#17334D'); ax.vlines(upper, y-.08, y+.08, color='#17334D')
ax.axvline(1, color='#8094A4', linestyle='--', linewidth=1.2)
ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 4.5)
ax.set_xlabel('Speedup vs fastest accepted FP64 control')
ax.set_title('Resident computation', loc='left', fontweight='bold', pad=16)
for i, r in enumerate(rows):
    ax.text(4.4, i, f"{r['speedup']:.2f}×", va='center', ha='right', fontweight='bold')
bx.barh(y, error, color='#17A398', height=.55)
bx.set_yticks(y, [''] * len(rows)); bx.invert_yaxis(); bx.set_xlim(0, 36)
bx.set_xlabel('Decrease in largest sampled Frobenius error (%)')
bx.set_title('Sampled numerical accuracy', loc='left', fontweight='bold', pad=16)
for i, r in enumerate(rows):
    bx.text(35.5, i, f"{r['worst_component_error_reduction_percent']:.4f}%", va='center', ha='right')
for axis in (ax, bx):
    axis.set_axisbelow(True); axis.grid(axis='x', alpha=.15)
fig.suptitle('K GraphFuse | Precision GPU portfolio', x=.045, y=.96, ha='left', fontsize=23, fontweight='bold', color='#17334D')
fig.text(.045, .905, 'Tesla T4 · tuned CuPy 14.2 / PyTorch 2.11 controls · 6 of 6 combined gates passed', fontsize=12, color='#536575')
fig.text(.045, .12, 'Timing: 5 blocks × 31 samples; bars: pooled median ratios; whiskers: 95% bootstrap of block medians.\n'
         'Accuracy: independent 384-bit reference on first 16 rows; values are already near the FP64 rounding floor.\n'
         'One constructed asymmetric graph family. Validation, transfers and JIT excluded. No native-cuGraph comparison.',
         fontsize=10, color='#536575', linespacing=1.6)
fig.subplots_adjust(left=.26, right=.96, top=.82, bottom=.24, wspace=.26)
out = base / 'figures'; out.mkdir(exist_ok=True)
for extension in ('png', 'svg'):
    fig.savefig(out / f'precision_t4_comparison.{extension}', dpi=200, facecolor='white')
plt.close(fig)
