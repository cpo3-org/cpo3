"""Build isolated variants and unpack the recorded local manifest; no checkout edits."""
from pathlib import Path
import argparse, gzip, io, os, shutil, subprocess, tarfile

parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[3])
parser.add_argument('--base', default='f937dabd37acfbc48ef04d285dd45bfb1c84aeb8')
parser.add_argument('--variants', default='baseline,candidate')
args = parser.parse_args()
archive = Path(__file__).resolve().parents[1]
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
for name in ['driver.py', 'worker.py', 'analyze.py', 'plot.py']:
    shutil.copy2(archive/'scripts'/name, out/name)
with gzip.open(archive/'results/suite.json.gz', 'rt') as f:
    (out/'suite.json').write_text(f.read())
# This is a trusted local Git archive. Use a new output directory for each run.
source = subprocess.check_output(['git', 'archive', args.base, 'Cargo.toml', 'Cargo.lock', 'src', 'python', 'tests'], cwd=args.repo)
env = dict(os.environ)
env.setdefault('CARGO_TARGET_DIR', str(out/'target'))
for variant in args.variants.split(','):
    assert variant in ['baseline', 'monotone', 'candidate', 'best_effort']
    root = out/(variant+'-source')
    root.mkdir()
    with tarfile.open(fileobj=io.BytesIO(source)) as tar:
        tar.extractall(root, filter='data')
    if variant != 'baseline':
        subprocess.run(['git', 'apply', str(archive/'patches'/(variant+'.patch'))], cwd=root, check=True)
    with (out/(variant+'-build.log')).open('w') as log:
        subprocess.run(['cargo', 'rustc', '--release', '--offline', '--locked', '--features', 'python,mps', '--manifest-path', str(root/'Cargo.toml'), '--lib', '--', '-C', 'link-arg=-undefined', '-C', 'link-arg=dynamic_lookup'], env=env, stdout=log, stderr=log, check=True)
    package = out/(variant+'-package')
    shutil.copytree(root/'python', package, ignore=shutil.ignore_patterns('__pycache__', '*.so'))
    shutil.copy2(Path(env['CARGO_TARGET_DIR'])/'release/libcpo3.dylib', package/'cpo3/_core.abi3.so')
    print('Built', variant, flush=True)
