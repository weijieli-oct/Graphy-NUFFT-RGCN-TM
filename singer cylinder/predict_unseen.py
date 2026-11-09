
import argparse
from copy import deepcopy
import json
from pathlib import Path
from params import latest_training_directory, ROOT, fresh_output_directory
from model import load_checkpoint
from solver import run

PREDICT_RE = [50., 70., 90., 110., 130., 150.]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(latest_training_directory()/'checkpoint.pt'))
    parser.add_argument('--re', type=float, nargs='+', default=PREDICT_RE)
    parser.add_argument('--config', help='Optional target mesh/config; architecture and gate/budget must match')
    parser.add_argument('--output', default=str(ROOT/'runs'/'unseen'))
    args = parser.parse_args()
    saved = load_checkpoint(args.checkpoint)
    base = deepcopy(saved['config'])
    if args.config:
        target = json.loads(Path(args.config).read_text(encoding='utf8'))
        for key in ('model', 'tolerances', 'loss', 'training', 'physics'):
            if target[key] != base[key]:
                raise ValueError(f'Controlled transfer requires identical {key}')
        base = target
    root = fresh_output_directory(args.output)
    failed = False
    for reynolds in args.re:
        if reynolds <= 0:
            raise ValueError('Re must be positive')
        if reynolds in saved['config']['cases']['re'] and not args.config:
            raise ValueError('Use a new Re or specify a target mesh for cross-mesh evaluation')
        c = deepcopy(base)
        c['cases'] = {'re': [reynolds], 'amplitude': [base['cases']['amplitude'][0]]}
        code, _ = run(c, root/f'Re_{reynolds:g}', initialization=saved)
        failed |= code != 0
    return 2 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
