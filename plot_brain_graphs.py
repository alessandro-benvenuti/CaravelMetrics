#!/usr/bin/env python3
"""
Simple viewer for brain vessel labels stored as directories containing:
 - vessel_graph_aligned.vtp
 - vessel_data.pkl

Usage examples:
  python3 plot_brain_graphs.py --input-dir ./Results --label CereVess_AB10520 --color red
  python3 plot_brain_graphs.py --input-dir ./Results --all --color rde
  python3 plot_brain_graphs.py --test    # runs an internal test (no external files needed)

Requirements:
  pip install vedo
"""

import argparse
import os
import sys
import json
import pickle
from pathlib import Path

try:
    from vedo import load, Plotter
except Exception as e:
    print('vedo is required. Install via: pip install vedo')
    raise


def find_label_folders(input_dir):
    p = Path(input_dir)
    if not p.exists():
        raise FileNotFoundError(input_dir)

    mesh_sources = []
    for child in sorted(p.iterdir()):
        if child.is_dir():
            vtps = sorted(child.glob('*.vtp'))
            for vtp in vtps:
                mesh_sources.append((vtp.stem, vtp))

    if mesh_sources:
        return mesh_sources

    for vtp in sorted(p.rglob('*.vtp')):
        mesh_sources.append((vtp.stem, vtp))
    return mesh_sources


def load_label_mesh(label_source):
    """Load a .vtp file from a folder or a direct file path."""
    source = Path(label_source)
    if source.is_file() and source.suffix.lower() == '.vtp':
        return load(str(source))

    vtp_path = source/'vessel_graph_aligned.vtp'
    if not vtp_path.exists():
        vtps = list(source.glob('*.vtp'))
        if len(vtps) == 0:
            return None
        vtp_path = vtps[0]
    mesh = load(str(vtp_path))
    return mesh


def resolve_colors(count, color_spec=None):
    palette = [
        'steelblue', 'tomato', 'seagreen', 'mediumorchid', 'goldenrod',
        'dodgerblue', 'crimson', 'darkorange', 'teal', 'slategray'
    ]

    if color_spec is None:
        return [palette[i % len(palette)] for i in range(count)]

    colors = [c.strip() for c in color_spec.split(',') if c.strip()]
    if not colors:
        return [palette[i % len(palette)] for i in range(count)]

    if len(colors) == 1:
        return colors * count

    return [colors[i % len(colors)] for i in range(count)]


def load_label_meta(label_folder):
    pkl_path = Path(label_folder)/'vessel_data.pkl'
    if not pkl_path.exists():
        pkls = list(Path(label_folder).glob('*.pkl'))
        if len(pkls)==0:
            return None
        pkl_path = pkls[0]
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f'Failed to load pickle {pkl_path}: {e}')
        return None


def show_single(label_folder, offscreen=False, save_path=None, color='steelblue'):
    mesh = load_label_mesh(label_folder)
    meta = load_label_meta(label_folder)
    label_name = Path(label_folder).name
    if meta is not None:
        print(f'Loaded meta for {label_name}: type={type(meta)}')
        # print summary keys if dict-like
        if isinstance(meta, dict):
            print('meta keys:', list(meta.keys()))
    else:
        print(f'No metadata (.pkl) found for {label_name}')

    if mesh is None:
        print('No .vtp mesh found in', label_folder)
        return

    try:
        mesh.c(color)
    except Exception:
        pass

    vp = Plotter(offscreen=offscreen, bg='white', size=(1000, 800))
    title = f'{label_name}'
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    vp.show(mesh, title=title, axes=1, zoom=1.0, interactive=not offscreen)
    if save_path:
        out = str(Path(save_path))
        vp.screenshot(out)
        print('Saved screenshot to', out)
    vp.close()


def show_all(input_dir, offscreen=False, save_path=None, color=None):
    mesh_sources = find_label_folders(input_dir)
    actors = []
    colors = resolve_colors(len(mesh_sources), color)

    for idx, (label_name, mesh_path) in enumerate(mesh_sources):
        mesh = load_label_mesh(mesh_path)
        if mesh is None:
            continue
        try:
            mesh.c(colors[idx])
        except Exception:
            pass
        mesh.name = label_name
        actors.append(mesh)

    if len(actors) == 0:
        print('No meshes found in', input_dir)
        return

    vp = Plotter(offscreen=offscreen, bg='white', size=(1400, 900))
    vp.show(actors, title='All regions', axes=1, interactive=not offscreen)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        vp.screenshot(str(save_path))
        print('Saved screenshot to', save_path)
    vp.close()


def run_test(offscreen=True):
    # create a synthetic polyline (tree-like) and display it
    import numpy as np
    from vedo import Line, Points
    pts1 = np.array([[0,0,0],[10,0,0],[20,5,0],[30,10,0]])
    pts2 = pts1 + np.array([0,20,0])
    l1 = Line(pts1).c('red').lw(4)
    l2 = Line(pts2).c('blue').lw(4)
    p1 = Points(pts1, r=6).c('darkred')
    p2 = Points(pts2, r=6).c('darkblue')
    vp = Plotter(offscreen=offscreen, bg='white', size=(900,700))
    vp.show([l1, l2, p1, p2], title='TEST SYNTHETIC', axes=1, interactive=not offscreen)
    out = 'test_synthetic_plot.png'
    vp.screenshot(out)
    print('Wrote', out)
    vp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-i', help='Directory containing label folders', default='brain_graphs/outputs')
    parser.add_argument('--label', '-l', help='Specific label folder name to show')
    parser.add_argument('--all', action='store_true', help='Show all labels in the input-dir')
    parser.add_argument('--save', '-s', help='Save screenshot to this path')
    parser.add_argument('--color', default=None, help='Color or comma-separated list of colors for the plotted graphs (e.g. red, blue, green, or steelblue,red,green)')
    parser.add_argument('--test', action='store_true', help='Run synthetic test (no files required)')
    parser.add_argument('--offscreen', action='store_true', help='Render offscreen (non-interactive)')
    args = parser.parse_args()

    if args.test:
        run_test(offscreen=True if args.offscreen or args.test else False)
        return

    if not Path(args.input_dir).exists():
        print('Input dir not found:', args.input_dir)
        sys.exit(1)

    if args.label and not args.all:
        label_folder = Path(args.input_dir)/args.label
        if not label_folder.exists():
            print('Label folder not found:', label_folder)
            sys.exit(1)
        show_single(label_folder, offscreen=args.offscreen, save_path=args.save, color=args.color)
    else:
        show_all(args.input_dir, offscreen=args.offscreen, save_path=args.save, color=args.color)


if __name__ == '__main__':
    main()
