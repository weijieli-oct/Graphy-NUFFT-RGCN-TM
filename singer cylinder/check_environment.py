

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', default='environment_report.json')
    args = parser.parse_args()
    modules = ('numpy', 'scipy.spatial', 'torch', 'finufft', 'cufinufft',
               'pytorch_finufft', 'matplotlib', 'pytest', 'gmsh')
    report = {'python_executable': sys.executable, 'python_version': sys.version,
              'platform': platform.platform(), 'requested_device': args.device,
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
            device = torch.device(args.device)
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
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report['passed']:
        print('Environment check failed. Keep your known-good environment; inspect the specific missing module/DLL above.', flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
