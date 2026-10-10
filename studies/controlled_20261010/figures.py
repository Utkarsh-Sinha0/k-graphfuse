"""Render controlled-study figures from the audited aggregate, including losses."""
from pathlib import Path
import hashlib, io, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ASSETS = ROOT / 'docs/assets'
S = json.loads((HERE/'SUMMARY.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'kgf-controlled-v1',
    'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#D6DFE7',
    'text.color':'#17243A','axes.labelcolor':'#17243A','xtick.color':'#5C6878','ytick.color':'#17243A'})
ARTIFACTS = []


def save(fig, name):
    for extension in ('png', 'svg'):
        path = ASSETS / (name+'.'+extension)
        if extension == 'svg':
            stream = io.StringIO();fig.savefig(stream,format='svg',bbox_inches='tight',metadata={'Date':None})
            path.write_text(stream.getvalue(),encoding='utf-8',newline='\n')
        else:fig.savefig(path,dpi=220,bbox_inches='tight',metadata={'Software':'K GraphFuse controlled study'})
        ARTIFACTS.append(path)
    plt.close(fig)


rows = S['packed_balanced'];labels = []
for r in rows[:10]:
    labels.append(f"{r['family'].replace('_',' ')} · {r['n']:,} × {r['q']}"+(' · derivative' if r.get('derivative') else ''))
fig, ax = plt.subplots(figsize=(12,7.8));y = np.arange(10)
for device, shift, color in [('RTX 3060 Laptop',-.17,'#6755C7'),('T4',.17,'#008C91')]:
    group = [r for r in rows if r['device']==device]
    values = np.array([r['time_reduction_pct'] for r in group])
    bounds = np.array([[100*(1-1/v) for v in r['ratio_ci95']] for r in group])
    ax.barh(y+shift,values,height=.31,color=color,label=device)
    ax.errorbar(values,y+shift,xerr=np.maximum(0,np.array([values-bounds[:,0],bounds[:,1]-values])),fmt='none',ecolor='#26344A',capsize=2,lw=.8)
    for i,v in enumerate(values):ax.text((max(v,bounds[i,1])+2) if v>=0 else (min(v,bounds[i,0])-2),i+shift,f'{v:+.1f}%',va='center',ha='left' if v>=0 else 'right',fontsize=8)
ax.set_yticks(y,labels);ax.invert_yaxis();ax.axvline(0,color='#8896A6',lw=.8)
ax.set_xlim(-70,103);ax.set_xlabel('Compute-time reduction vs fastest accepted packed + balanced CuPy control (%)')
ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.legend(loc='lower right',frameon=False)
fig.suptitle('K GraphFuse | matching budgets and retained degrees',x=.025,ha='left',fontweight='bold',fontsize=18)
fig.text(.025,.91,'18/20 pooled device–fixture comparisons pass the strong speed gate; R-MAT loses on both devices.',fontsize=11)
fig.text(.025,.015,'Frozen repaired implementation · fresh seeds 20261013/14 · 5 × 11 samples per seed · FP64\n'
    'Whiskers: descriptive paired block 95% intervals. Validated resident compute; excludes checks, setup and transfers.\n'
    'Six synthetic families; 10 fixtures × 2 devices. This is not native cuGraph or exhaustive SOTA.',fontsize=9,color='#5C6878')
fig.subplots_adjust(left=.35,right=.96,top=.87,bottom=.16)
save(fig,'controlled_matched_degrees')

fig, axes = plt.subplots(1,2,figsize=(11.6,5.5),gridspec_kw={'width_ratios':[1.25,1]})
app = S['application'];x = np.arange(2)
axes[0].bar(x-.18,[r['control_ms'] for r in app],.34,label='Specialized CuPy restart',color='#6755C7')
axes[0].bar(x+.18,[r['k_ms'] for r in app],.34,label='K GraphFuse (checked)',color='#008C91')
for i,r in enumerate(app):
    axes[0].text(i-.18,r['control_ms']+7,f"{r['control_ms']:.1f}",ha='center')
    axes[0].text(i+.18,r['k_ms']+7,f"{r['k_ms']:.1f}",ha='center')
axes[0].set_xticks(x,['RTX 3060 Laptop','Tesla T4']);axes[0].set_ylim(0,465)
axes[0].set_ylabel('Complete 256-user batch latency (ms)');axes[0].legend(frameon=False,loc='upper left',fontsize=9)
axes[1].axis('off');axes[1].text(.03,.95,'Recommendation quality matches\nthe CPU reference',fontsize=12,fontweight='bold',va='top')
axes[1].text(.03,.73,'Recall@10       0.058742\nnDCG@10        0.094498\nTop-10 lists     100% exact agreement',fontsize=12,linespacing=1.8,va='top')
axes[1].text(.03,.32,'RTX: K uses 7.13% more median time.\nT4: +0.07% time reduction is a practical tie.\nBoth ratio intervals include one.\nNo demonstrated end-to-end median gain.',fontsize=11,linespacing=1.6,color='#5C6878',va='top')
fig.suptitle('K GraphFuse | real-data application control',x=.025,ha='left',fontweight='bold',fontsize=18)
fig.text(.025,.01,'MovieLens 1M · chronological holdout · fixed finite graph diffusion, not learned recommendation SOTA\n'
    'Includes seed construction, transfer, finite checks, output and ranking. Pooled medians of 10 complete batches/device;\n'
    'setup and initial compilation are separate. All timed blocks, including first-use outliers, remain in the raw record.',fontsize=9,color='#5C6878')
fig.subplots_adjust(top=.86,bottom=.2,wspace=.24,left=.08,right=.97)
save(fig,'controlled_application')
record = {'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [HERE/'SUMMARY.json',Path(__file__)]},
    'artifact_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in ARTIFACTS}}
(HERE/'VISUALS.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
assert len(ARTIFACTS)==4 and all(p.stat().st_size>10000 for p in ARTIFACTS)
print('PASS: two controlled-study figures, PNG/SVG, with negative results and exact source hashes.')
