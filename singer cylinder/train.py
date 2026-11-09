
import argparse
from copy import deepcopy
import json
import params as settings
from model import load_checkpoint
from solver import run


def train(config=None, output=None, resume=None, initialize=None):
    resume = resume or settings.RESUME_CHECKPOINT
    if resume and initialize:
        raise ValueError('Choose resume OR shared initialization')
    saved = load_checkpoint(settings.resolve_path(resume)) if resume else None
    if saved is not None:
        if config is not None:
            raise ValueError('Exact resume uses the saved config; use --initialize for new cases/meshes')
        c = deepcopy(saved['config'])
        out = settings.resolve_path(output) if output else settings.resolve_path(resume).parent
    else:
        c = deepcopy(config or settings.CONFIG)
        out = settings.fresh_output_directory(output or settings.TRAIN_OUTPUT)
    initialization = load_checkpoint(settings.resolve_path(initialize)) if initialize else None
    result = run(c, out, saved=saved, initialization=initialization)
    settings.remember_training_directory(out)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Experiment JSON configuration')
    parser.add_argument('--output')
    parser.add_argument('--resume')
    parser.add_argument('--initialize', help='Load only compatible v2 model weights; reset state/replay/optimizer')
    args = parser.parse_args()
    c = json.loads(settings.resolve_path(args.config).read_text(encoding='utf8')) if args.config else None
    code, out = train(c, args.output, args.resume, args.initialize)
    print('结果目录:', out)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
