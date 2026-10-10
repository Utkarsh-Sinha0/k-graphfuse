"""Render every matched bound-constant timing pair, without inferred intervals."""
from pathlib import Path
import hashlib,io,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
S=json.loads((HERE/'SUMMARY.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'kgf-bound-attribution-v1',
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,
    'text.color':'#17243A','axes.labelcolor':'#17243A','xtick.color':'#5C6878','ytick.color':'#17243A'})
fig,axes=plt.subplots(1,2,figsize=(12,5.8))
for ax,group,title in zip(axes,[S['matched_fp64_pairs'][:4],S['matched_fp64_pairs'][4:]],
    ['100,000 vertices × 4 channels\nPolynomial · degree 23 → 22',
     '1,000,000 vertices × 8 channels\nValue + derivative · degree 19 → 18']):
    y=np.arange(4);old=np.array([r['old_median_ms'] for r in group]);new=np.array([r['new_median_ms'] for r in group])
    ax.barh(y-.18,old,height=.32,label='Older bound: 1 + √2',color='#6755C7')
    ax.barh(y+.18,new,height=.32,label='New bound: 2',color='#008C91')
    labels=[f"Packed {'yes' if r['packed'] else 'no'} · capture {'yes' if r['capture'] else 'no'}" for r in group]
    ax.set_yticks(y,labels);ax.invert_yaxis();ax.set_xlim(0,float(old.max())*1.43)
    for i,r in enumerate(group):
        ax.text(old[i]+old.max()*.02,i-.18,f'{old[i]:.3f}',va='center',fontsize=8)
        ax.text(new[i]+old.max()*.02,i+.18,f'{new[i]:.3f}  (−{r["time_reduction_pct"]:.2f}%)',va='center',fontsize=8)
    ax.set_title(title,loc='left',fontweight='bold',fontsize=12,pad=18)
    ax.set_xlabel('Observed median resident compute time (ms)');ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
fig.suptitle('K GraphFuse | isolating the theorem-bound constant',x=.025,ha='left',fontweight='bold',fontsize=18)
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.53,.92),ncol=2,frameon=False,fontsize=10)
fig.text(.025,.02,'Same stored operator, inputs, coefficients, error budget, FP64 backend and execution settings within each pair.\n'
    '2/11 legacy records reduce degree · Tesla T4 · 31 samples/configuration · no repeat-run interval available.\n'
    '3.56–5.35% less observed median time across all eight pairs; these are below the separate 10% strong-speed gate.\n'
    'Original degree-9 → 8 boundary is untimed. Laptop constant comparisons retain equal degrees.',fontsize=9,color='#5C6878')
fig.subplots_adjust(left=.13,right=.97,top=.70,bottom=.24,wspace=.52)
paths=[]
for extension in ('png','svg'):
    p=ROOT/f'docs/assets/bound_attribution.{extension}'
    if extension=='svg':
        stream=io.StringIO();fig.savefig(stream,format='svg',bbox_inches='tight',metadata={'Date':None})
        p.write_text(stream.getvalue(),encoding='utf-8',newline='\n')
    else:fig.savefig(p,dpi=220,bbox_inches='tight')
    paths.append(p)
plt.close(fig)
record={'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'SUMMARY.json',Path(__file__))},
    'artifact_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
(HERE/'VISUALS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
assert len(paths)==2 and all(p.stat().st_size>10000 for p in paths)
print('Rendered all eight bound-attribution pairs with explicit uncertainty and laptop limits.')
