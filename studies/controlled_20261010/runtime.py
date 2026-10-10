"""Load the toolkit-selected NVRTC before PyTorch; preserve the frozen method."""
from pathlib import Path
import hashlib, json, runpy, sys
import cupy as cp

here=Path(__file__).resolve().parent
version=cp.cuda.nvrtc.getVersion()
if '--output' in sys.argv:
    output=Path(sys.argv[sys.argv.index('--output')+1]);output.parent.mkdir(parents=True,exist_ok=True)
    record=dict(nvrtc=list(version),cupy_accelerators='default CUB enabled',
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Compatibility preload only; no accelerator disabled and no frozen algorithm or selection policy changed.')
    output.with_suffix('.runtime.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8',newline='\n')
print('Toolkit-selected NVRTC preloaded:',version,flush=True)
runpy.run_path(str(here/'study.py'),run_name='__main__')
