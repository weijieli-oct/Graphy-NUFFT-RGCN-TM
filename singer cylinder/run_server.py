
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import server_settings as settings

ROOT = Path(__file__).resolve().parent


def local(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT/p


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf8')


def command(logs, label, arguments):
    argv = [sys.executable, '-u', *[str(v) for v in arguments]]
    print('\n>>>', subprocess.list2cmdline(argv), flush=True)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    with (logs/(label+'.log')).open('w', encoding='utf8') as log:
        with subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, encoding='utf8', errors='replace') as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise
    if code:
        raise subprocess.CalledProcessError(code, argv)


def configuration(experiment, device, logs):
    from params import CONFIG
    if experiment == 'airfoil_default':
        c = deepcopy(CONFIG)
    else:
        folder = ROOT/'experiments'
        command(logs, 'experiment_configs', ['experiments.py', '--output', folder, '--device', 'cuda'])
        filename = folder/(experiment+'.json')
        if not filename.exists():
            raise ValueError(f'Unknown experiment: {experiment}; see server_settings.py')
        c = json.loads(filename.read_text(encoding='utf8'))
    c['device'] = device
    c['verification_report'] = str(ROOT/'operator_validation.json')
    if settings.RE_LIST is not None:
        if not settings.RE_LIST or min(settings.RE_LIST) <= 0:
            raise ValueError('RE_LIST must contain positive values')
        c['cases'] = {'re': list(settings.RE_LIST), 'amplitude': [settings.AMPLITUDE]*len(settings.RE_LIST)}
    for key, value in settings.TRAINING_OVERRIDES.items():
        if key not in c['training']:
            raise ValueError(f'Unknown training option: {key}')
        c['training'][key] = value
    return c


def prepare(c, config_path, logs):
    command(logs, 'environment', ['check_environment.py', '--device', c['device'],
                                 '--output', logs/'environment.json'])
    command(logs, 'operator_validation', ['sanity_check.py', '--device', c['device'],
                                         '--output', c['verification_report']])
    command(logs, 'regressions', ['-m', 'pytest', 'test_solver.py', 'test_evaluate.py', '-q',
                                 '--junitxml='+str(logs/'pytest_report.xml')])
    mesh = Path(c['grid']['mesh_path']); report = mesh.with_suffix('.json')
    if settings.REBUILD_MESH or not mesh.exists() or not report.exists():
        command(logs, 'mesh', ['mesh.py', '--config', config_path])
    else:
        q = json.loads(report.read_text(encoding='utf8'))
        if q.get('geometry') != c['geometry']:
            raise ValueError('Existing mesh geometry differs from config. Set REBUILD_MESH=True to regenerate.')
        print('Reusing existing mesh:', mesh, flush=True)
    from model import validate
    validate(c)
    if settings.SAVE_MESH_FIGURES:
        command(logs, 'mesh_figures', ['view_mesh.py', '--config', config_path])


def summarize(folder, logs, label):
    status_file = folder/'status.json'
    if not status_file.exists():
        raise FileNotFoundError(status_file)
    status = json.loads(status_file.read_text(encoding='utf8'))
    if status.get('status') != 'complete':
        raise RuntimeError(f'Solve did not complete: {status_file}')
    if settings.STATISTICS_START_TIME is not None:
        command(logs, label, ['evaluate.py', '--history', folder/'history.csv',
                             '--start-time', settings.STATISTICS_START_TIME,
                             '--output', folder/'statistics.json'])


def predict(checkpoint_path, output, logs, transfer=None):
    from model import load_checkpoint
    saved = load_checkpoint(checkpoint_path)
    trained = saved['config']['cases']['re']
    values = settings.UNSEEN_RE
    if values is None:
        values = [v for v in (50., 70., 90., 110., 130., 150.) if v not in trained]
    if not values:
        raise ValueError('Specify UNSEEN_RE (for cross-mesh, training Re values are also allowed)')
    argv = ['predict_unseen.py', '--checkpoint', checkpoint_path, '--output', output,
            '--re', *values]
    if transfer:
        argv += ['--config', local(transfer)]
    command(logs, 'unseen_solves', argv)
    for reynolds in values:
        summarize(output/f'Re_{reynolds:g}', logs, f'statistics_Re_{reynolds:g}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['prepare', 'all', 'train', 'resume', 'predict'], default=settings.ACTION)
    parser.add_argument('--experiment', default=settings.EXPERIMENT)
    parser.add_argument('--device', default=settings.DEVICE)
    args = parser.parse_args()
    session = local(settings.RUNS_DIRECTORY)/(args.experiment+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    logs = session/'logs'; logs.mkdir(parents=True, exist_ok=False)
    write_json(session/'launcher.json', {'action': args.action, 'experiment': args.experiment,
                'device': args.device, 'interpreter': sys.executable, 'project': ROOT,
                'run_unseen_after_train': settings.RUN_UNSEEN_AFTER_TRAIN,
                'statistics_start_time': settings.STATISTICS_START_TIME})
    print('Session folder:', session, flush=True)
    try:
        if args.action == 'resume':
            if settings.RESUME_CHECKPOINT is None:
                raise ValueError('Set RESUME_CHECKPOINT in server_settings.py')
            cp = local(settings.RESUME_CHECKPOINT)
            command(logs, 'resume', ['train.py', '--resume', cp])
            summarize(cp.parent, logs, 'statistics')
        elif args.action == 'predict':
            if settings.SOURCE_CHECKPOINT is None:
                raise ValueError('Set SOURCE_CHECKPOINT in server_settings.py')
            predict(local(settings.SOURCE_CHECKPOINT), session/'unseen', logs, settings.TRANSFER_CONFIG)
        else:
            c = configuration(args.experiment, args.device, logs)
            cfg = session/'config_requested.json'; write_json(cfg, c)
            if args.action in ('prepare', 'all'):
                prepare(c, cfg, logs)
            if args.action in ('train', 'all'):
                trained = session/'train'
                argv = ['train.py', '--config', cfg, '--output', trained]
                if settings.INITIALIZE_CHECKPOINT:
                    argv += ['--initialize', local(settings.INITIALIZE_CHECKPOINT)]
                command(logs, 'training', argv)
                summarize(trained, logs, 'statistics')
                if args.action == 'all' and settings.RUN_UNSEEN_AFTER_TRAIN:
                    predict(trained/'checkpoint.pt', session/'unseen', logs)
        write_json(session/'launcher_status.json', {'status': 'complete', 'physical_validation': False})
        print('Requested stages complete:', session, flush=True)
        return 0
    except subprocess.CalledProcessError as exc:
        write_json(session/'launcher_status.json', {'status': 'stopped', 'returncode': exc.returncode,
                   'command': exc.cmd, 'note': 'Inspect stage logs; no later stage was started.'})
        print('Stopped at a failed stage. Logs:', logs, flush=True)
        return exc.returncode
    except KeyboardInterrupt:
        write_json(session/'launcher_status.json', {'status': 'interrupted'})
        print('Interrupted. Resume from the last completed checkpoint.', flush=True)
        return 130
    except Exception as exc:
        write_json(session/'launcher_status.json', {'status': 'error', 'error': str(exc)})
        raise


if __name__ == '__main__':
    raise SystemExit(main())
