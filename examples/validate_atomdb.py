"""Run reproducible validation against an existing local AtomDB directory.

Run with PyAtomDB and this package installed:
  python examples/validate_atomdb.py --root /path/to/APED --output validation.json
This runner needs the tests/ directory from the source checkout.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import unittest


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(8*1024*1024):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('atomdb-validation.json'))
    args = parser.parse_args()
    root = args.root.resolve()
    os.environ['SHERPA_PLASMA_ATOMDB'] = str(root)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    import test_atomdb_real
    files = (os.environ.get('SHERPA_PLASMA_ATOMDB_LINEFILE', 'apec_line.fits'),
             os.environ.get('SHERPA_PLASMA_ATOMDB_COCOFILE', 'apec_coco.fits'), 'filemap', 'VERSION')
    before = {name: digest(root/name) for name in files}
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name)
        for name in ('test_plasma_integration', 'test_atomdb_real'))
    started = perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = {name: digest(root/name) for name in files}
    report = {'database_version': (root/'VERSION').read_text().strip(),
        'database_sha256': before, 'database_inputs_unchanged': before == after,
        'versions': {name: importlib.metadata.version(name) for name in
                     ('pyatomdb', 'astropy', 'numpy')},
        'python': platform.python_version(), 'seconds': perf_counter()-started,
        'tests_run': result.testsRun, 'failures': [(str(t), text) for t, text in result.failures],
        'errors': [(str(t), text) for t, text in result.errors],
        'skipped': [(str(t), reason) for t, reason in result.skipped],
        'metrics': test_atomdb_real.METRICS,
        'validation': 'PASS' if result.wasSuccessful() and not result.skipped and before == after else 'FAIL'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('Validation report:', args.output, flush=True)
    raise SystemExit(0 if report['validation'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
