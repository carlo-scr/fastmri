"""Generate data_acquisition_colab.ipynb (run: python build_notebook.py).

The notebook is self-contained: it clones nothing, so it can be pasted into any
Colab session. Raw data goes to the runtime's local disk (/content/raw), only
packed tars and checkpoints are copied to Google Drive, and nothing touches the Mac.
"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# MIDL 2027 (FA-KGD): data acquisition

Fetches every dataset and checkpoint the paper needs **beyond** the brain cohort, which is
already packed as `midl27_brain.tar`. Raw downloads land on the Colab runtime disk
(`/content/raw`, deleted with the runtime). Only small **packed** tars go to Drive, in the same
format as `experiments/midl27/pack_data.py`: central slices, original attributes, plus
`midl27_source_slices`, and gzip-compressed k-space.

| Step | What | Paper use | Download | On Drive (approx.) |
|---|---|---|---|---|
| A | fastMRI+ boxes (CSV) | Tab. 6 pathology ROI | <10 MB | <10 MB |
| B | knee EDM prior (InverseBench `MRI-knee.pt`) + knee E2E-VarNet | Tab. 3 knee, C8 | 0.5 GB | 0.5 GB |
| C | M4Raw 0.3 T: test (25 subjects) + val (calibration subjects) | Tab. 5, endpoint E3 | 8 GB | ~2–4 GB |
| D | fastMRI knee multi-coil **val** (needs your NYU licence link) | Tab. 3 knee, Tab. 6 | streamed, stops early | ~3–5 GB |
| E | SHA-256 manifest of everything on Drive | pre-registration (Tab. 1 caption) | — | — |

Runtime: **CPU is enough** (no GPU needed). Use High-RAM if available.
Every step is idempotent: if its output already exists on Drive it is skipped, so re-running
after a disconnect is safe. Check the Drive quota printed by the first cell before step C/D.
""")

code(r"""
# 0  config: edit these, then Runtime -> Run all
ART = '/content/drive/MyDrive/fastmri_artifacts'   # same folder the experiment book reads

DO_FASTMRI_PLUS = True
DO_CHECKPOINTS  = True
DO_M4RAW        = True
DO_KNEE         = True

# M4Raw (Zenodo 10.5281/zenodo.8056074, CC BY 4.0)
M4RAW_SLICES        = 8      # central slices kept per file (files have ~18)
M4RAW_TEST_SUBJECTS = None   # None = all 25 test subjects (E3 unit of analysis is the subject)
M4RAW_CALIB_SUBJECTS = 4     # subjects taken from the val split, for tuning only

# fastMRI knee multi-coil val. Paste the 'knee_multicoil_val' link from the NYU licence
# e-mail (https://fastmri.med.nyu.edu). The link expires; request a new one if it 403s.
KNEE_URL       = ''
KNEE_N_VOLUMES = 28          # mirrors the brain cohort: 4 calibration + 24 test
KNEE_SLICES    = 8
KNEE_SELECT    = 'boxes'     # 'boxes': only volumes with fastMRI+ boxes; 'reviewed': any reviewed; 'any'
KNEE_ACQ       = None        # e.g. 'CORPD_FBK' to restrict to PD; None keeps PD and PDFS

SEED = 2027
""")

code(r"""
# 0.1  Drive + quota
from google.colab import drive
drive.mount('/content/drive')
import os, shutil
os.makedirs(ART, exist_ok=True)
RAW = '/content/raw'; os.makedirs(RAW, exist_ok=True)
tot, used, free = shutil.disk_usage('/content/drive/MyDrive')
print(f'Drive: {used/2**30:.1f} / {tot/2**30:.1f} GB used, {free/2**30:.1f} GB free')
tot, used, free = shutil.disk_usage('/content')
print(f'Runtime disk: {free/2**30:.1f} GB free')
for f in sorted(os.listdir(ART)):
    p = os.path.join(ART, f)
    if os.path.isfile(p):
        print(f'  {f:40s} {os.path.getsize(p)/2**30:6.2f} GB')
""")

code(r"""
# 0.2  helpers (pure Python; also unit-tested locally from build_notebook.py)
import os, re, io, json, glob, time, random, hashlib, tarfile, subprocess, collections
import numpy as np, h5py

def fetch(url, dst):
    # resumable download to the runtime disk; returns dst
    if os.path.exists(dst) and not os.path.exists(dst + '.part'):
        return dst
    open(dst + '.part', 'w').close()
    subprocess.run(['wget', '-q', '--show-progress', '-c', '-O', dst, url], check=True)
    os.remove(dst + '.part')
    return dst

def central_slices(nz, n):
    lo = max(0, nz // 2 - n // 2)
    return list(range(lo, min(nz, lo + n)))

def pack_h5(src, dst, n_slices, extra=None):
    # copy central slices of 'kspace' (+ attrs) into a gzip-compressed h5, like pack_data.py
    with h5py.File(src, 'r') as h:
        k = h['kspace']
        idx = central_slices(k.shape[0], n_slices)
        with h5py.File(dst, 'w') as o:
            o.create_dataset('kspace', data=k[idx], compression='gzip', compression_opts=4,
                             chunks=(1,) + k.shape[1:])
            if 'ismrmrd_header' in h:
                o.create_dataset('ismrmrd_header', data=h['ismrmrd_header'][()])
            for a, v in h.attrs.items():
                o.attrs[a] = v
            o.attrs['midl27_source_slices'] = json.dumps(idx)
            for a, v in (extra or {}).items():
                o.attrs[a] = v
        return dict(file=os.path.basename(dst), shape=list(k.shape), slices=idx)

def write_tar(cohort_dir, out_tar, meta):
    # tar the cohort folder (arcname = folder name) next to a PACKING.json
    with open(os.path.join(cohort_dir, 'PACKING.json'), 'w') as fp:
        json.dump(meta, fp, indent=1)
    tmp = out_tar + '.tmp'
    with tarfile.open(tmp, 'w') as t:
        t.add(cohort_dir, arcname=os.path.basename(cohort_dir))
    os.replace(tmp, out_tar)
    return out_tar

def sha256(path, chunk=2**24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()

def to_drive(local, name=None):
    # copy a finished artefact to Drive (atomic rename) and record its hash
    dst = os.path.join(ART, name or os.path.basename(local))
    digest = sha256(local)
    shutil.copy(local, dst + '.tmp'); os.replace(dst + '.tmp', dst)
    print(f'-> {dst}  ({os.path.getsize(dst)/2**30:.2f} GB, sha256 {digest[:12]}…)')
    return dst, digest

M4RAW_RE = re.compile(r'^(?P<subject>\d+)_(?P<contrast>T1|T2|FLAIR)(?P<rep>\d+)\.h5$')

def m4raw_parse(path):
    m = M4RAW_RE.match(os.path.basename(path))
    return None if m is None else dict(m.groupdict(), rep=int(m['rep']))

def pick_subjects(files, n, seed):
    # all files of n randomly chosen subjects (None = all), deterministic in seed
    by = collections.defaultdict(list)
    for f in files:
        p = m4raw_parse(f)
        if p: by[p['subject']].append(f)
    subs = sorted(by)
    if n is not None:
        subs = sorted(random.Random(seed).sample(subs, min(n, len(subs))))
    return subs, [f for s in subs for f in sorted(by[s])]

def stream_select_h5(fileobj, keep, n_max, out_dir, n_slices, acquisition=None):
    # Read a (possibly compressed) tar from a stream, pack the members that keep(basename)
    # accepts (and whose 'acquisition' attr matches, if given), and stop after n_max:
    # the rest of the archive is never downloaded.
    packed = []
    with tarfile.open(fileobj=fileobj, mode='r|*') as t:
        for m in t:
            name = os.path.basename(m.name)
            if not (m.isfile() and name.endswith('.h5') and not name.startswith('._') and keep(name)):
                continue
            tmp = os.path.join(out_dir, '_' + name)
            with t.extractfile(m) as src, open(tmp, 'wb') as dst:
                shutil.copyfileobj(src, dst, 2**24)
            try:
                if acquisition:
                    with h5py.File(tmp, 'r') as h:
                        if h.attrs.get('acquisition') != acquisition:
                            os.remove(tmp); continue
                rec = pack_h5(tmp, os.path.join(out_dir, name), n_slices)
            except OSError as e:          # truncated / unreadable member: skip, keep going
                print('  skip', name, e); rec = None
            os.remove(tmp)
            if rec:
                packed.append(rec); print(f'  packed {len(packed)}/{n_max}: {name}')
            if len(packed) >= n_max:
                break
    return packed

DONE = lambda name: os.path.exists(os.path.join(ART, name))
""")

md(r"""
## A — fastMRI+ pathology boxes
Brain and knee box CSVs plus the lists of reviewed files (a reviewed file without boxes had no
findings). Coverage of the packed brain cohort is printed so the test manifest can be chosen
with box coverage in mind **before** Stage 1 of the experiment book freezes it.
Note from fastMRI+: boxes are in the coordinates of the 320×320 reconstruction and must be
flipped vertically to overlay on fastMRI images (see their example notebook).
""")

code(r"""
# A  fastMRI+
FMP = f'{ART}/fastmri_plus'
if DO_FASTMRI_PLUS:
    os.makedirs(FMP, exist_ok=True)
    base = 'https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations'
    for f in ['brain.csv', 'brain_file_list.csv', 'knee.csv', 'knee_file_list.csv']:
        if not os.path.exists(f'{FMP}/{f}'):
            subprocess.run(['wget', '-q', '-O', f'{FMP}/{f}', f'{base}/{f}'], check=True)
    import pandas as pd
    def boxes(anat):
        d = pd.read_csv(f'{FMP}/{anat}.csv')
        return d[d['x'].notna()]
    def reviewed(anat):
        return set(pd.read_csv(f'{FMP}/{anat}_file_list.csv', header=None)[0])
    for anat in ['brain', 'knee']:
        b = boxes(anat)
        print(f'{anat}: {len(b)} boxes in {b.file.nunique()} files; {len(reviewed(anat))} files reviewed')

    # coverage of the brain cohort already on Drive (reads only PACKING.json from the tar)
    if DONE('midl27_brain.tar'):
        with tarfile.open(f'{ART}/midl27_brain.tar') as t:
            meta = json.load(t.extractfile('midl27_cohort/PACKING.json'))
        b = boxes('brain'); n_slices = meta['n_slices']
        rows = []
        for f in meta['files']:
            stem = f[:-3]; fb = b[b.file == stem]
            rows.append(dict(file=stem, boxes=len(fb), slices_with_boxes=sorted(fb.slice.unique().tolist())))
        cov = pd.DataFrame(rows)
        print(f"\nmidl27_brain.tar: {(cov.boxes > 0).sum()}/{len(cov)} volumes have boxes "
              f"(any slice; the tar keeps the central {n_slices} slices of each volume)")
        display(cov[cov.boxes > 0])
""")

md(r"""
## B — checkpoints
* `MRI-knee.pt`: InverseBench EDM prior (DhariwalUNet, 2 channels, **320×320 MVUE** with
  ESPIRiT maps), trained on fastMRI knee multi-coil **train**. Knee **val** is therefore a clean
  test split. Loading it needs the InverseBench model code (`models.precond.EDMPrecond`),
  not the Ambient-DPS pickle loader. That is integration work for the experiment book.
* `knee_leaderboard_state_dict.pt`: official fastMRI E2E-VarNet (supervised reference row).
* `MRI-brain.pt` is optional: an alternative brain prior (also 320×320), useful as a
  robustness row. Off by default.
""")

code(r"""
# B  checkpoints
CK = f'{ART}/checkpoints'
GET_INVERSEBENCH_BRAIN = False
if DO_CHECKPOINTS:
    os.makedirs(CK, exist_ok=True)
    want = {
        'inversebench_MRI-knee.pt': 'https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/MRI-knee.pt',
        'varnet_knee_leaderboard_state_dict.pt': 'https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/knee_leaderboard_state_dict.pt',
    }
    if GET_INVERSEBENCH_BRAIN:
        want['inversebench_MRI-brain.pt'] = 'https://github.com/devzhk/InverseBench/releases/download/diffusion-prior/MRI-brain.pt'
    for name, url in want.items():
        if os.path.exists(f'{CK}/{name}'):
            print('have', name); continue
        to_drive(fetch(url, f'{RAW}/{name}'), f'checkpoints/{name}')
    # the knee prior's architecture config, so it can be rebuilt without the repo
    cfg = f'{CK}/inversebench_mri-knee-mvue.yaml'
    if not os.path.exists(cfg):
        subprocess.run(['wget', '-q', '-O', cfg, 'https://raw.githubusercontent.com/devzhk/InverseBench/main/configs/pretrain/mri-knee-mvue.yaml'], check=True)
""")

md(r"""
## C — M4Raw (0.3 T, 4-channel brain)
`M4Raw_multicoil_test.zip` (5.0 GB): 25 new subjects, T1w ×6, T2w ×6 and FLAIR ×4 repetitions,
for endpoint E3. `M4RawV1.5_multicoil_val.zip` (3.0 GB) provides the **calibration** subjects
(tuning only). Files are named `<subject>_<contrast><rep>.h5`, with `kspace` stored as
(slice, coil, H, W) like fastMRI. **All repetitions of a chosen subject are kept**, because the
averaged reference in Tab. 5 needs them.

The inspection cell checks something our pipeline assumes: whether the readout is
oversampled, so that Ψ can be estimated from signal-free k-space rows. The M4Raw tutorial
shows 256×256 k-space, so this may **not** hold. Read the printout before using these tars.
""")

code(r"""
# C.1  download + unzip (runtime disk only)
M4 = f'{RAW}/m4raw'
M4_URL = 'https://zenodo.org/records/8056074/files/{}?download=1'
if DO_M4RAW and not (DONE('m4raw_test.tar') and DONE('m4raw_calib.tar')):
    os.makedirs(M4, exist_ok=True)
    for z, sub in [('M4Raw_multicoil_test.zip', 'test'), ('M4RawV1.5_multicoil_val.zip', 'val')]:
        d = f'{M4}/{sub}'
        if not glob.glob(f'{d}/**/*.h5', recursive=True):
            zp = fetch(M4_URL.format(z), f'{M4}/{z}')
            os.makedirs(d, exist_ok=True)
            subprocess.run(['unzip', '-q', '-o', zp, '-d', d], check=True)
            os.remove(zp)                                   # free runtime disk
        n = len(glob.glob(f'{d}/**/*.h5', recursive=True))
        print(f'{sub}: {n} h5 files')
""")

code(r"""
# C.2  inspect one file per contrast: layout, attrs, readout oversampling / noise-only rows
if DO_M4RAW and os.path.isdir(f'{M4}/test'):
    files = sorted(glob.glob(f'{M4}/test/**/*.h5', recursive=True))
    print('unparsed names:', [os.path.basename(f) for f in files if not m4raw_parse(f)][:5])
    seen = set()
    for f in files:
        p = m4raw_parse(f)
        if not p or p['contrast'] in seen: continue
        seen.add(p['contrast'])
        with h5py.File(f, 'r') as h:
            k = h['kspace']; ks = k[k.shape[0] // 2]                    # (C, H, W)
            print(f"\n{os.path.basename(f)}  keys={list(h.keys())}  kspace{k.shape} {k.dtype}")
            print('  attrs:', {a: (v if np.size(v) < 5 else '...') for a, v in h.attrs.items()})
            img = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(ks, axes=(-2, -1)), axes=(-2, -1)), axes=(-2, -1))
            rss = np.sqrt((np.abs(img) ** 2).sum(0))
            H, W = rss.shape; e = max(4, H // 16)
            edge = np.concatenate([rss[:e], rss[-e:]]).mean() / rss.mean()
            zero_cols = int((np.abs(ks).sum((0, 1)) == 0).sum())
            print(f'  image-edge/mean (readout ends) = {edge:.3f}  (<<1 suggests oversampled FOV with empty margins)')
            print(f'  all-zero k-space columns: {zero_cols}/{W}')
            if 'ismrmrd_header' in h:
                hdr = h['ismrmrd_header'][()].decode(errors='ignore')
                for tag in ['encodedSpace', 'reconSpace']:
                    m = re.search(f'<{tag}>.*?<matrixSize>(.*?)</matrixSize>', hdr, re.S)
                    if m: print(f'  {tag} matrix:', re.sub(r'\s+', ' ', m.group(1)))
""")

code(r"""
# C.3  pack: test (all or M4RAW_TEST_SUBJECTS subjects) and calibration (val split)
if DO_M4RAW:
    for split, src, n_sub, out in [('test', 'test', M4RAW_TEST_SUBJECTS, 'm4raw_test.tar'),
                                   ('calib', 'val', M4RAW_CALIB_SUBJECTS, 'm4raw_calib.tar')]:
        if DONE(out):
            print('have', out); continue
        files = sorted(glob.glob(f'{M4}/{src}/**/*.h5', recursive=True))
        subs, chosen = pick_subjects(files, n_sub, SEED)
        cohort = f'{RAW}/m4raw_{split}_cohort'; os.makedirs(cohort, exist_ok=True)
        recs = []
        for f in chosen:
            recs.append(dict(pack_h5(f, f'{cohort}/{os.path.basename(f)}', M4RAW_SLICES), **m4raw_parse(f)))
        meta = dict(src=f'zenodo:8056074/{src}', split=split, seed=SEED, n_slices=M4RAW_SLICES,
                    subjects=subs, files=[r['file'] for r in recs], records=recs)
        tar = write_tar(cohort, f'{RAW}/{out}', meta)
        print(f'{split}: {len(subs)} subjects, {len(recs)} files')
        to_drive(tar)
        shutil.rmtree(cohort)
""")

md(r"""
## D — fastMRI knee multi-coil val (streamed)
The NYU archive is read **as a stream**: matching volumes are packed as they arrive, and the
download stops once `KNEE_N_VOLUMES` are packed, so neither the runtime nor Drive ever holds
the full archive. With `KNEE_SELECT='boxes'`, only volumes with fastMRI+ boxes are kept, so the
same knee cohort serves Tab. 3 and Tab. 6. That makes it a pathology-enriched cohort; say so
in the setup section. Selection follows archive order, which is deterministic for a given archive.
""")

code(r"""
# D  knee (needs KNEE_URL)
if DO_KNEE and not DONE('midl27_knee.tar'):
    assert KNEE_URL, 'paste the knee_multicoil_val link from the NYU licence e-mail into KNEE_URL (cell 0)'
    import urllib.request, pandas as pd
    FMP = f'{ART}/fastmri_plus'
    kb = pd.read_csv(f'{FMP}/knee.csv'); kb = set(kb[kb['x'].notna()].file)
    kr = set(pd.read_csv(f'{FMP}/knee_file_list.csv', header=None)[0])
    pool = {'boxes': kb, 'reviewed': kr, 'any': None}[KNEE_SELECT]
    keep = lambda name: pool is None or name[:-3] in pool
    cohort = f'{RAW}/midl27_knee_cohort'; os.makedirs(cohort, exist_ok=True)
    t0 = time.time()
    req = urllib.request.Request(KNEE_URL, headers={'User-Agent': 'Wget/1.21'})
    with urllib.request.urlopen(req) as resp:
        recs = stream_select_h5(resp, keep, KNEE_N_VOLUMES, cohort, KNEE_SLICES, KNEE_ACQ)
    print(f'packed {len(recs)} knee volumes in {(time.time() - t0) / 60:.1f} min')
    meta = dict(src='fastMRI knee_multicoil_val (stream)', acquisition=KNEE_ACQ, select=KNEE_SELECT,
                n_slices=KNEE_SLICES, files=[r['file'] for r in recs], records=recs)
    to_drive(write_tar(cohort, f'{RAW}/midl27_knee.tar', meta))
    shutil.rmtree(cohort)
""")

md(r"""
## E — manifest
Writes `DATA_MANIFEST.json` on Drive with the size and SHA-256 of every tar and checkpoint.
Commit a copy to the repo (`MIDL2027/`) **before** any test run; it is the frozen data manifest
the paper's Tab. 1 caption promises. Hashing re-reads files from Drive, so it takes a few minutes.
""")

code(r"""
# E  manifest
import datetime
entries = {}
for p in sorted(glob.glob(f'{ART}/*.tar') + glob.glob(f'{ART}/*.pkl') + glob.glob(f'{ART}/checkpoints/*')):
    rel = os.path.relpath(p, ART)
    entries[rel] = dict(bytes=os.path.getsize(p), sha256=sha256(p))
    print(f"{rel:50s} {entries[rel]['bytes']/2**30:6.2f} GB  {entries[rel]['sha256'][:12]}…")
man = dict(created=datetime.datetime.utcnow().isoformat() + 'Z', files=entries)
with open(f'{ART}/DATA_MANIFEST.json', 'w') as fp:
    json.dump(man, fp, indent=1)
print('wrote', f'{ART}/DATA_MANIFEST.json')
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 0}
out = Path(__file__).with_name("data_acquisition_colab.ipynb")
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(cells)} cells)")
