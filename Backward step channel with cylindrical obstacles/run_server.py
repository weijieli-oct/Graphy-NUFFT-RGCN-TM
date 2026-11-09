
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
    """Log output and stop on any failure, including residual-gate exit code 2."""
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


def check_environment(device_name, output):
    import importlib
    import importlib.metadata
    import platform

    modules = ('numpy', 'scipy.spatial', 'torch', 'finufft', 'cufinufft',
               'pytorch_finufft', 'matplotlib', 'pytest', 'gmsh')
    report = {'python_executable': sys.executable, 'python_version': sys.version,
              'platform': platform.platform(), 'requested_device': device_name,
              'packages': {}, 'errors': [], 'numerical_verification': 'not_performed_by_this_script'}
    for name in modules:
        try:
            module = importlib.import_module(name)
            package = name.split('.')[0].replace('_', '-')
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, '__version__', 'unknown/local build')
            report['packages'][name] = {'version': version, 'path': getattr(module, '__file__', None)}
        except Exception as exc:
            report['errors'].append(f'{name}: {type(exc).__name__}: {exc}')
    if 'torch' in report['packages']:
        import torch
        report['torch_cuda_runtime'] = torch.version.cuda
        report['cuda_available'] = torch.cuda.is_available()
        try:
            device = torch.device(device_name)
            if device.type != 'cuda' or not torch.cuda.is_available():
                raise RuntimeError('Select your previous working CUDA interpreter in PyCharm')
            props = torch.cuda.get_device_properties(device)
            report['gpu'] = {'name': props.name, 'memory_GiB': props.total_memory/1024**3,
                             'compute_capability': [props.major, props.minor]}
            from nufft_compat import finufft_type1, finufft_type2
            report['nufft_wrapper_imported'] = callable(finufft_type1) and callable(finufft_type2)
        except Exception as exc:
            report['errors'].append(f'CUDA/wrapper: {type(exc).__name__}: {exc}')
    report['passed'] = not report['errors']
    target = Path(output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report['passed']:
        print('Environment check failed. Keep your known-good environment; inspect the specific missing module/DLL above.', flush=True)
    return 0 if report['passed'] else 1


def configuration(experiment, device, logs):
    from params import configuration as case_configuration
    c = case_configuration(experiment, device)
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
    command(logs, 'environment', ['run_server.py', '--check-environment', '--device', c['device'],
                                 '--output', logs/'environment.json'])

    command(logs, 'operator_validation', ['sanity_check.py', '--device', c['device'],
                                         '--output', c['verification_report']])
    command(logs, 'regressions', ['-m', 'pytest', 'tests', '-q',
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
    command(logs, 'case_validation', ['verify_case.py', '--config', config_path,
                                      '--device', c['device'], '--output', logs/'case_validation.json'])
    if settings.SAVE_MESH_FIGURES:
        command(logs, 'mesh_figures', ['mesh.py', '--view', '--config', config_path])


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
                             '--body-history', folder/'body_history.csv',
                             '--output', folder/'statistics.json'])
    if settings.PLOT_AFTER_SOLVE:
        command(logs, label+'_figures', ['visualize_results.py', '--run', folder])


def predict(checkpoint_path, output, logs, transfer=None):
    from model import load_checkpoint
    saved = load_checkpoint(checkpoint_path)
    trained = saved['config']['cases']['re']
    values = settings.UNSEEN_RE
    if values is None:
        values = [v for v in (50., 70., 90., 110., 130., 150.) if v not in trained]
    if not values:
        raise ValueError('Specify UNSEEN_RE (for cross-mesh, training Re values are also allowed)')
    argv = ['solver.py', 'predict', '--checkpoint', checkpoint_path, '--output', output,
            '--re', *values]
    if transfer:
        argv += ['--config', local(transfer)]
    command(logs, 'unseen_solves', argv)
    for reynolds in values:
        summarize(output/f'Re_{reynolds:g}', logs, f'statistics_Re_{reynolds:g}')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['mesh', 'prepare', 'all', 'train', 'resume', 'predict'], default=settings.ACTION)
    parser.add_argument('--experiment', default=settings.EXPERIMENT)
    parser.add_argument('--device', default=settings.DEVICE)
    parser.add_argument('--check-environment', action='store_true', help='Inspect environment only')
    parser.add_argument('--output', default=str(ROOT/'environment_report.json'),
                        help='JSON output for --check-environment')
    args = parser.parse_args(argv)
    if args.check_environment:
        return check_environment(args.device, args.output)
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
            command(logs, 'resume', ['solver.py', 'train', '--resume', cp])
            summarize(cp.parent, logs, 'statistics')
        elif args.action == 'predict':
            if settings.SOURCE_CHECKPOINT is None:
                raise ValueError('Set SOURCE_CHECKPOINT in server_settings.py')
            predict(local(settings.SOURCE_CHECKPOINT), session/'unseen', logs, settings.TRANSFER_CONFIG)
        else:
            c = configuration(args.experiment, args.device, logs)
            cfg = session/'config_requested.json'; write_json(cfg, c)
            if args.action == 'mesh':
                command(logs, 'mesh', ['mesh.py', '--config', cfg])
                command(logs, 'mesh_figures', ['mesh.py', '--view', '--config', cfg])
            if args.action in ('prepare', 'all'):
                prepare(c, cfg, logs)
            if args.action in ('train', 'all'):
                trained = session/'train'
                argv = ['solver.py', 'train', '--config', cfg, '--output', trained]
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
