from __future__ import annotations
import argparse, os, sys, time
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
SLICE_ID         = "Q49-EMBEDDING-EXPORT-PIPELINE"
FEAT_DIM_RAW     = 960
BACKBONE_OUT_DIM = 128
SEED             = 42
CLAHE_CLIP       = 3.0
CLAHE_TILE       = (4, 4)
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]
DATASET_ROOT     = "data/processed/vindr_binary_roi_224"
OUTPUT_DIR       = "workspace/experiments/Q49/embeddings"
REPORT_PATH      = "workspace/experiments/Q49/embedding_export_report.md"
BATCH_SIZE       = 32
NUM_WORKERS      = 0
SPLITS           = ["train", "val", "test"]

def build_extractor(device):
    import torch.nn as nn
    from torchvision import models
    full_model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    backbone = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    projection = nn.Linear(FEAT_DIM_RAW, BACKBONE_OUT_DIM)
    for p in projection.parameters():
        p.requires_grad = False
    projection.eval()
    return backbone.to(device), projection.to(device)

def export_split(split, ds_root, backbone, projection, device, output_dir):
    import numpy as np, torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
    from run_q38a_preprocessing_benchmark import _clahe_torch
    transform = T.Compose([
        T.Lambda(lambda x: _clahe_torch(x, CLAHE_CLIP, CLAHE_TILE)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = VinDrSpineXRBinaryDataset(root=ds_root, split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=False, prefetch_factor=None)
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = projection(backbone(imgs))
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(labels.numpy())
    embeddings_np = np.concatenate(all_emb, axis=0)
    labels_np = np.concatenate(all_lbl, axis=0)
    out_emb = os.path.join(output_dir, split + "_embeddings.npy")
    out_lbl = os.path.join(output_dir, split + "_labels.npy")
    np.save(out_emb, embeddings_np)
    np.save(out_lbl, labels_np)
    return len(labels_np), int((labels_np == 0).sum()), int((labels_np == 1).sum())

def build_extractor(device):
    import torch.nn as nn
    from torchvision import models
    full_model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
    backbone = nn.Sequential(full_model.features, full_model.avgpool, nn.Flatten())
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    projection = nn.Linear(FEAT_DIM_RAW, BACKBONE_OUT_DIM)
    for p in projection.parameters():
        p.requires_grad = False
    projection.eval()
    return backbone.to(device), projection.to(device)

def export_split(split, ds_root, backbone, projection, device, output_dir):
    import numpy as np, torch
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
    from run_q38a_preprocessing_benchmark import _clahe_torch
    transform = T.Compose([
        T.Lambda(lambda x: _clahe_torch(x, CLAHE_CLIP, CLAHE_TILE)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    dataset = VinDrSpineXRBinaryDataset(root=ds_root, split=split, transform=transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True,
                        persistent_workers=False, prefetch_factor=None)
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            emb = projection(backbone(imgs))
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(labels.numpy())
    embeddings_np = np.concatenate(all_emb, axis=0)
    labels_np = np.concatenate(all_lbl, axis=0)
    np.save(os.path.join(output_dir, split + '_embeddings.npy'), embeddings_np)
    np.save(os.path.join(output_dir, split + '_labels.npy'), labels_np)
    return len(labels_np), int((labels_np == 0).sum()), int((labels_np == 1).sum())

def write_report(report_path, split_stats, elapsed_s, device_str):
    bt = chr(96)
    all_pass = all(
        split_stats[s][3] == '(' + str(split_stats[s][0]) + ', ' + str(BACKBONE_OUT_DIM) + ')'
        for s in SPLITS if s in split_stats
    )
    rows = []
    for split in SPLITS:
        if split not in split_stats: continue
        n, c0, c1, shape = split_stats[split]
        pos_pct = 100.0 * c1 / n if n > 0 else 0.0
        rows.append('| ' + split + ' | ' + str(n) + ' | ' + str(c0) + ' | ' + str(c1) + ' | ' + ('%.1f' % pos_pct) + '% | ' + shape + ' |')
    dim_rows = []
    for split in SPLITS:
        if split not in split_stats: continue
        n, c0, c1, shape = split_stats[split]
        expected = '(' + str(n) + ', ' + str(BACKBONE_OUT_DIM) + ')'
        status = 'PASS' if shape == expected else 'FAIL (expected ' + expected + ')'
        dim_rows.append('- **' + split + '**: shape=' + shape + ' -> ' + status)
    file_rows = []
    out_dir = os.path.join(_PROJECT_ROOT, OUTPUT_DIR)
    for split in SPLITS:
        for suffix in ['embeddings', 'labels']:
            fpath = os.path.join(out_dir, split + '_' + suffix + '.npy')
            if os.path.isfile(fpath):
                size_mb = os.path.getsize(fpath) / 1024 / 1024
                rel = os.path.relpath(fpath, _PROJECT_ROOT)
                file_rows.append("- " + bt + rel + bt + " (" + ("%.3f" % size_mb) + " MB)")
    dim_status = chr(65)+chr(76)+chr(76)+chr(32)+chr(80)+chr(65)+chr(83)+chr(83) if all_pass else chr(70)+chr(65)+chr(73)+chr(76)+chr(85)+chr(82)+chr(69)+chr(83)
    shape_status = 'PASS' if all_pass else 'FAIL'
    date_str = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
    proj_str = 'Linear(' + str(FEAT_DIM_RAW) + ' -> ' + str(BACKBONE_OUT_DIM) + '), frozen, seed=' + str(SEED)
    clahe_str = 'CLAHE(clip=' + str(CLAHE_CLIP) + ', tile=' + str(CLAHE_TILE) + ') + 3ch + ImageNet norm'
    header = [
        '# Q49 Embedding Export Report', '',
        '**Slice**: ' + SLICE_ID + '  ',
        '**Date**: ' + date_str + '  ',
        '**Backbone**: MobileNetV3-Large (IMAGENET1K_V1, frozen)  ',
        '**Projection**: ' + proj_str + '  ',
        '**Embedding dim**: ' + str(BACKBONE_OUT_DIM) + '  ',
        '**Device**: ' + device_str + '  ',
        '**Wall time**: ' + ('%.1f' % elapsed_s) + 's  ',
        '', '## Split Summary', '',
        '| Split | N | Class 0 | Class 1 | pos% | Shape |',
        '|-------|---|---------|---------|------|-------|',
    ]
    footer = [
        '', '## Dimensionality Verification', '',
    ] + dim_rows + [
        '', '**Check**: ' + dim_status,
        '', '## Output Files', '',
    ] + file_rows + [
        '', '## Validation', '',
        '- shape (N,' + str(BACKBONE_OUT_DIM) + '): ' + shape_status,
        '- Backbone frozen: True',
        '- Preprocessing: ' + clahe_str,
        '- shuffle=False',
    ]
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as fh:
        fh.write('\n'.join(header + rows + footer) + '\n')
    print('  Report: ' + report_path)

def run_dry_run(project_root):
    ok = True
    print('=' * 70)
    print(SLICE_ID + ' -- DRY-RUN')
    print('=' * 70)
    ds_path = os.path.join(project_root, DATASET_ROOT)
    for split in SPLITS:
        sp = os.path.join(ds_path, split)
        if os.path.isdir(sp):
            n = sum(len(os.listdir(os.path.join(sp, c))) for c in ['0', '1'] if os.path.isdir(os.path.join(sp, c)))
            print('  [OK] dataset/' + split + ': ' + str(n) + ' files')
        else:
            print('  [MISSING] dataset/' + split)
            ok = False
    for pkg in ['torch', 'torchvision', 'numpy']:
        try:
            __import__(pkg)
            print('  [OK] ' + pkg)
        except ImportError as e:
            print('  [MISSING] ' + pkg + ': ' + str(e))
            ok = False
    try:
        from qcore.data.vindr_spinexr import VinDrSpineXRBinaryDataset
        print('  [OK] qcore')
    except ImportError as e:
        print('  [MISSING] qcore: ' + str(e))
        ok = False
    out_path = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(out_path, exist_ok=True)
    print('  [OK] output dir: ' + out_path)
    print('  Result: ' + ('READY' if ok else 'BLOCKED'))
    sys.exit(0 if ok else 1)

def main():
    parser = argparse.ArgumentParser(description=SLICE_ID)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--project-root', default=_PROJECT_ROOT)
    args = parser.parse_args()
    project_root = os.path.abspath(args.project_root)
    if args.dry_run:
        run_dry_run(project_root)
    try:
        import torch
        import numpy as np
    except ImportError as e:
        print('ERROR: ' + str(e) + ' -- run inside docker-qstrata-gpu-1')
        sys.exit(1)
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n' + SLICE_ID)
    print('=' * 70)
    print('  Backbone:    mobilenet_v3_large (frozen, IMAGENET1K_V1)')
    print('  Projection:  ' + str(FEAT_DIM_RAW) + ' -> ' + str(BACKBONE_OUT_DIM))
    print('  Device:      ' + str(device))
    backbone, projection = build_extractor(device)
    print('  Backbone params:   ' + str(sum(p.numel() for p in backbone.parameters())))
    print('  Projection params: ' + str(sum(p.numel() for p in projection.parameters())))
    ds_root = os.path.join(project_root, DATASET_ROOT)
    output_dir = os.path.join(project_root, OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)
    split_stats = {}
    t_start = time.time()
    for split in SPLITS:
        print('  [' + split + '] exporting...', end='', flush=True)
        ts = time.time()
        n, c0, c1 = export_split(split, ds_root, backbone, projection, device, output_dir)
        dt = time.time() - ts
        emb = np.load(os.path.join(output_dir, split + '_embeddings.npy'))
        shape_str = str(emb.shape)
        split_stats[split] = (n, c0, c1, shape_str)
        pos_pct = 100.0 * c1 / n if n > 0 else 0.0
        print('  shape=' + shape_str + '  c0=' + str(c0) + '  c1=' + str(c1) + '  pos=' + ('%.1f' % pos_pct) + '%  ' + ('%.1f' % dt) + 's')
    elapsed_s = time.time() - t_start
    print('\n  Total: ' + ('%.1f' % elapsed_s) + 's')
    dim_ok = all(
        split_stats[s][3] == '(' + str(split_stats[s][0]) + ', ' + str(BACKBONE_OUT_DIM) + ')'
        for s in SPLITS if s in split_stats
    )
    print('  Dimensionality (N,' + str(BACKBONE_OUT_DIM) + '): ' + ('PASS' if dim_ok else 'FAIL'))
    report_path = os.path.join(project_root, REPORT_PATH)
    write_report(report_path, split_stats, elapsed_s, str(device))
    total_written = sum(1 for s in SPLITS if os.path.isfile(
        os.path.join(output_dir, s + '_embeddings.npy')))
    report_exists = os.path.isfile(report_path)
    passed = total_written == 3 and dim_ok and report_exists
    print('\nPASS/FAIL Checklist:')
    print('  [' + ('OK' if total_written == 3 else 'FAIL') + '] train/val/test .npy written (' + str(total_written) + '/3)')
    print('  [' + ('OK' if dim_ok else 'FAIL') + '] shape (N,' + str(BACKBONE_OUT_DIM) + ') verified')
    print('  [' + ('OK' if report_exists else 'FAIL') + '] export report written')
    print('\n  RESULT: ' + ('PASS' if passed else 'FAIL'))
    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
