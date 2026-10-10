"""Fresh-seed paired confirmation against packed, balanced framework controls."""
from pathlib import Path
import argparse, json
import study as s

HERE=Path(__file__).resolve().parent
METHODS=['cupy_onepass_packed','cupy_iterated_packed','fused_onepass_captured']
SEEDS=[20261013,20261014]


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--seed',type=int,default=SEEDS[0])
    p.add_argument('--output',type=Path);args=p.parse_args()
    if args.freeze:
        record=dict(study_id='kgf-packed-balanced-controls-20261010-v1',methods=METHODS,seeds=SEEDS,
            runner_sha256=s.digest(Path(__file__)),base_freeze_sha256=s.digest(HERE/'FREEZE.json'),
            purpose='Stronger combined packing/balancing controls, identified during study review; frozen before fresh-seed confirmation. Same graphs, budgets and backends; retain all outcomes.')
        (HERE/'CONTROL_FREEZE.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n');return
    lock=json.loads((HERE/'CONTROL_FREEZE.json').read_text(encoding='utf-8'))
    assert s.digest(Path(__file__))==lock['runner_sha256'] and s.check_freeze()==lock['base_freeze_sha256']
    assert args.seed in SEEDS and args.output is not None
    import cupy as cp
    compiler=cp.cuda.nvrtc.getVersion()
    import torch
    torch.set_num_threads(1);torch.cuda.init()
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    s.PROTOCOL['methods']=METHODS
    report=dict(study_id=lock['study_id'],freeze_sha256=s.digest(HERE/'CONTROL_FREEZE.json'),base_freeze_sha256=s.digest(HERE/'FREEZE.json'),
        source_sha256=s.sources(),seed=args.seed,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cupy=cp.__version__,nvrtc=list(compiler),status='running',cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
    try:s.ablations(args.seed,report,save);report['status']='complete'
    except Exception as error:report['status']='failed';report['error']=type(error).__name__;raise
    finally:save()


if __name__=='__main__':main()
