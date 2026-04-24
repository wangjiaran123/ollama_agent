import os
import sys
import json
import uuid
import shutil
import subprocess
import re
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

SANDBOX_DIR = os.path.join(os.path.dirname(__file__), 'sandbox_output')
os.makedirs(SANDBOX_DIR, exist_ok=True)

def _build_restricted_script(code: str, output_dir: str) -> str:
    clean_code = _remove_dangerous_code(code)
    safe_open_source = """
import sys, json, base64, traceback, io, contextlib, os
sys.path.insert(0, OUTPUT_DIR)
os.chdir(OUTPUT_DIR)

_output_capture = {'stdout': '', 'stderr': '', 'files': [], 'plots': [], 'error': None}
_stdout_buffer = io.StringIO()
_stderr_buffer = io.StringIO()

def _save_output_file(filename, data_bytes, is_plot=False):
    filepath = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(data_bytes)
    entry = {'filename': filename, 'is_plot': is_plot}
    if is_plot:
        _output_capture['plots'].append(entry)
    else:
        _output_capture['files'].append(entry)

def _safe_open(file, mode='r', *args, **kwargs):
    allowed_read_exts = {'.txt', '.csv', '.json', '.xml', '.yaml', '.yml', '.md', '.log',
                         '.html', '.css', '.js', '.py', '.jpg', '.jpeg', '.png', '.gif',
                         '.bmp', '.webp', '.svg', '.ico', '.xlsx', '.xls', '.pdf'}
    allowed_write_exts = {'.txt', '.csv', '.json', '.xml', '.yaml', '.yml', '.md', '.log',
                          '.html', '.css', '.js', '.py', '.jpg', '.jpeg', '.png', '.gif',
                          '.bmp', '.webp', '.svg', '.html'}
    if isinstance(file, int):
        raise PermissionError('file descriptor access is not allowed')
    fname = str(file)
    ext = os.path.splitext(fname)[1].lower()
    if '..' in fname.split(os.sep):
        raise PermissionError('directory traversal is not allowed')
    if 'w' in mode or 'a' in mode or 'x' in mode:
        if ext and ext not in allowed_write_exts:
            raise PermissionError(f'writing .{ext} files is not allowed')
    else:
        if ext and ext not in allowed_read_exts:
            raise PermissionError(f'reading .{ext} files is not allowed')
    return open(file, mode, *args, **kwargs)

_BLOCKED_MODULES = {
    'os', 'subprocess', 'shutil', 'signal', 'ctypes',
    'socket', 'sys', 'multiprocessing', 'threading',
    'importlib', 'pickle', 'shelve', 'marshal',
    'pty', 'tty', 'termios', 'fcntl',
    'code', 'codeop', 'codecs',
    'crypt', 'grp', 'pwd', 'spwd',
    'cProfile', 'profile', 'trace',
    'webbrowser', 'antigravity',
    'distutils', 'http.server', 'http.client',
    'urllib.request', 'urllib.response', 'urllib.parse',
    'xmlrpc', 'ftplib', 'smtplib', 'poplib', 'imaplib',
    'telnetlib', 'paramiko', 'pexpect',
    'winreg', 'win32api', 'win32pipe',
}

def _safe_import(name, *args, **kwargs):
    if name in _BLOCKED_MODULES:
        raise ImportError(f'module {name} is not allowed in sandbox')
    return __import__(name, *args, **kwargs)

def _capture_plt():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        _plot_counter = [0]
        def _patched_show():
            _plot_counter[0] += 1
            fname = f'plot_{_plot_counter[0]}.png'
            plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=100, bbox_inches='tight')
            _save_output_file(fname, open(os.path.join(OUTPUT_DIR, fname), 'rb').read(), is_plot=True)
            plt.close()
        plt.show = _patched_show
    except ImportError:
        pass

_capture_plt()

_safe_builtins = {
    '__import__': _safe_import,
    'True': True, 'False': False, 'None': None,
    'print': print, 'len': len, 'range': range, 'int': int, 'float': float,
    'str': str, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
    'set': set, 'abs': abs, 'max': max, 'min': min, 'sum': sum,
    'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
    'zip': zip, 'map': map, 'filter': filter, 'any': any, 'all': all,
    'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr,
    'round': round, 'pow': pow, 'hex': hex, 'oct': oct, 'bin': bin,
    'ord': ord, 'chr': chr, 'type': type,
    'open': _safe_open,
    'iter': iter, 'next': next, 'slice': slice,
    'property': property, 'staticmethod': staticmethod, 'classmethod': classmethod,
    'super': super, 'object': object, 'Exception': Exception,
    'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'ImportError': ImportError, 'ZeroDivisionError': ZeroDivisionError,
    'FileNotFoundError': FileNotFoundError, 'StopIteration': StopIteration,
}

try:
    import numpy as _np
    _safe_builtins['numpy'] = _np
    _safe_builtins['np'] = _np
except ImportError:
    pass

try:
    import pandas as _pd
    _safe_builtins['pandas'] = _pd
    _safe_builtins['pd'] = _pd
except ImportError:
    pass

try:
    from PIL import Image as _PILImage
    _safe_builtins['PIL'] = type(sys)('PIL')
    _safe_builtins['PIL'].Image = _PILImage
    _safe_builtins['Image'] = _PILImage
except ImportError:
    pass

try:
    import math as _math
    _safe_builtins['math'] = _math
except ImportError:
    pass

try:
    import random as _random
    _safe_builtins['random'] = _random
except ImportError:
    pass

try:
    import datetime as _dt
    _safe_builtins['datetime'] = _dt
except ImportError:
    pass

try:
    import collections as _collections
    _safe_builtins['collections'] = _collections
except ImportError:
    pass

try:
    import itertools as _itertools
    _safe_builtins['itertools'] = _itertools
except ImportError:
    pass

try:
    import functools as _functools
    _safe_builtins['functools'] = _functools
except ImportError:
    pass

try:
    import json as _json
    _safe_builtins['json'] = _json
except ImportError:
    pass

try:
    import re as _re
    _safe_builtins['re'] = _re
except ImportError:
    pass

try:
    import statistics as _stats
    _safe_builtins['statistics'] = _stats
except ImportError:
    pass

try:
    import csv as _csv
    _safe_builtins['csv'] = _csv
except ImportError:
    pass

try:
    import hashlib as _hashlib
    _safe_builtins['hashlib'] = _hashlib
except ImportError:
    pass

try:
    import base64 as _b64
    _safe_builtins['base64'] = _b64
except ImportError:
    pass

try:
    import io as _io
    _safe_builtins['io'] = _io
except ImportError:
    pass

try:
    import copy as _copy
    _safe_builtins['copy'] = _copy
except ImportError:
    pass

try:
    import pathlib as _pathlib
    _safe_builtins['pathlib'] = _pathlib
except ImportError:
    pass

try:
    import decimal as _decimal
    _safe_builtins['decimal'] = _decimal
except ImportError:
    pass

try:
    import fractions as _fractions
    _safe_builtins['fractions'] = _fractions
except ImportError:
    pass

try:
    import string as _string
    _safe_builtins['string'] = _string
except ImportError:
    pass

try:
    import textwrap as _textwrap
    _safe_builtins['textwrap'] = _textwrap
except ImportError:
    pass

try:
    import pprint as _pprint
    _safe_builtins['pprint'] = _pprint
except ImportError:
    pass

try:
    import enum as _enum
    _safe_builtins['enum'] = _enum
except ImportError:
    pass

try:
    import dataclasses as _dataclasses
    _safe_builtins['dataclasses'] = _dataclasses
except ImportError:
    pass

try:
    import operator as _operator
    _safe_builtins['operator'] = _operator
except ImportError:
    pass

try:
    import typing as _typing
    _safe_builtins['typing'] = _typing
except ImportError:
    pass

try:
    from sklearn import neighbors, cluster, preprocessing, model_selection, metrics as _skm
    _safe_builtins['sklearn'] = type(sys)('sklearn')
except ImportError:
    pass

try:
    import scipy as _scipy
    _safe_builtins['scipy'] = _scipy
except ImportError:
    pass

try:
    import seaborn as _sns
    _safe_builtins['seaborn'] = _sns
except ImportError:
    pass

try:
    import plotly as _plotly
    _safe_builtins['plotly'] = _plotly
except ImportError:
    pass

try:
    with contextlib.redirect_stdout(_stdout_buffer), contextlib.redirect_stderr(_stderr_buffer):
        _exec_globals = {'__builtins__': _safe_builtins}
        exec(USER_CODE, _exec_globals)
except SystemExit:
    pass
except Exception as e:
    _output_capture['error'] = traceback.format_exc()

_output_capture['stdout'] = _stdout_buffer.getvalue()
_output_capture['stderr'] = _stderr_buffer.getvalue()
print('<<<SANDBOX_RESULT>>>')
print(json.dumps(_output_capture))
"""
    output_dir_escaped = output_dir.replace('\\', '\\\\').replace("'", "\\'")
    safe_open_source = safe_open_source.replace('OUTPUT_DIR', repr(output_dir))
    safe_open_source = safe_open_source.replace('USER_CODE', repr(clean_code))
    return safe_open_source


def _remove_dangerous_code(code: str) -> str:
    dangerous_patterns = [
        (r'__import__\s*\(', 'cannot use __import__'),
        (r'__builtins__', 'cannot access __builtins__'),
        (r'__class__', 'cannot access __class__'),
        (r'__subclasses__', 'cannot access __subclasses__'),
        (r'__globals__', 'cannot access __globals__'),
        (r'exec\s*\(', 'exec() is not allowed'),
        (r'eval\s*\(', 'eval() is not allowed'),
        (r'compile\s*\(', 'compile() is not allowed'),
        (r'os\.system', 'os.system is not allowed'),
        (r'os\.popen', 'os.popen is not allowed'),
        (r'subprocess\.', 'subprocess is not allowed'),
        (r'ctypes\.', 'ctypes is not allowed'),
        (r'socket\.', 'socket is not allowed'),
        (r'open\s*\(\s*0\b', 'file descriptor access is not allowed'),
    ]
    clean = code
    for pattern, msg in dangerous_patterns:
        if re.search(pattern, clean):
            clean = re.sub(pattern, f'# BLOCKED: {msg} - ', clean)
    return clean


class SandboxResult:
    def __init__(self, success: bool, stdout: str = '', stderr: str = '',
                 files: List[Dict] = None, plots: List[Dict] = None,
                 error: str = None, execution_time: float = 0.0):
        self.success = success
        self.stdout = stdout
        self.stderr = stderr
        self.files = files or []
        self.plots = plots or []
        self.error = error
        self.execution_time = execution_time

    def to_dict(self) -> Dict:
        return {
            'success': self.success,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'files': self.files,
            'plots': self.plots,
            'error': self.error,
            'execution_time': round(self.execution_time, 3),
        }


class SandboxExecutor:
    def __init__(self):
        self.execution_dir = SANDBOX_DIR
        self.timeout = 30
        self.max_output_size = 10 * 1024 * 1024
        self._running_processes: Dict[str, subprocess.Popen] = {}

    def execute(self, code: str, session_id: str = None, timeout: int = None, tags: List[str] = None) -> SandboxResult:
        exec_id = str(uuid.uuid4())[:8]
        work_dir = os.path.join(self.execution_dir, f'run_{exec_id}')
        os.makedirs(work_dir, exist_ok=True)

        if tags and 'cache' in tags:
            try:
                with open(os.path.join(work_dir, '.cache_tag'), 'w') as f:
                    f.write('cache')
            except Exception:
                pass

        timeout = timeout or self.timeout
        start_time = datetime.now()

        try:
            restricted_script = _build_restricted_script(code, work_dir)
            script_path = os.path.join(work_dir, '_sandbox_script.py')
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(restricted_script)

            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            env['MPLBACKEND'] = 'Agg'

            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
                env=env,
                cwd=work_dir,
            )

            tracking_key = f'{session_id or "global"}:{exec_id}'
            self._running_processes[tracking_key] = proc

            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                elapsed = (datetime.now() - start_time).total_seconds()
                return SandboxResult(
                    success=False,
                    stdout='',
                    stderr='',
                    error=f'Execution timed out after {timeout} seconds',
                    execution_time=elapsed,
                )
            finally:
                self._running_processes.pop(tracking_key, None)

            elapsed = (datetime.now() - start_time).total_seconds()
            stdout_str = stdout_bytes.decode('utf-8', errors='replace')
            stderr_str = stderr_bytes.decode('utf-8', errors='replace')

            result_data = self._parse_result(stdout_str)
            if result_data:
                output_files = self._collect_output_files(work_dir, result_data)
                plots = result_data.get('plots', [])
                for p in plots:
                    p['url'] = f'/api/code/output/{exec_id}/{p["filename"]}'
                files = result_data.get('files', [])
                for f in files:
                    f['url'] = f'/api/code/output/{exec_id}/{f["filename"]}'

                return SandboxResult(
                    success=result_data['error'] is None,
                    stdout=result_data.get('stdout', ''),
                    stderr=result_data.get('stderr', ''),
                    files=files,
                    plots=plots,
                    error=result_data.get('error'),
                    execution_time=elapsed,
                )
            else:
                return SandboxResult(
                    success=False,
                    stdout=stdout_str,
                    stderr=stderr_str,
                    error='Failed to parse sandbox output',
                    execution_time=elapsed,
                )

        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f'Sandbox execution error: {e}')
            return SandboxResult(
                success=False,
                error=f'Internal sandbox error: {str(e)}',
                execution_time=elapsed,
            )

    def stop_execution(self, session_id: str):
        for key, proc in list(self._running_processes.items()):
            if key.startswith(f'{session_id}:'):
                try:
                    proc.kill()
                except Exception:
                    pass
                del self._running_processes[key]

    def get_output_file(self, exec_id: str, filename: str) -> Optional[str]:
        filepath = os.path.join(self.execution_dir, f'run_{exec_id}', filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            return filepath
        return None

    def cleanup_session(self, session_id: str, max_age_hours: int = 24):
        import time
        now = time.time()
        for item in os.listdir(self.execution_dir):
            item_path = os.path.join(self.execution_dir, item)
            if os.path.isdir(item_path) and item.startswith('run_'):
                age_hours = (now - os.path.getctime(item_path)) / 3600
                if age_hours > max_age_hours:
                    try:
                        shutil.rmtree(item_path)
                    except Exception as e:
                        logger.error(f'Cleanup error for {item_path}: {e}')

    def _parse_result(self, stdout: str) -> Optional[Dict]:
        marker = '<<<SANDBOX_RESULT>>>'
        if marker in stdout:
            json_str = stdout.split(marker, 1)[1].strip()
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
        return None

    def _collect_output_files(self, work_dir: str, result_data: Dict) -> List[Dict]:
        all_files = []
        for entry in result_data.get('files', []):
            fname = entry.get('filename', '')
            fpath = os.path.join(work_dir, fname)
            if os.path.exists(fpath):
                entry['size'] = os.path.getsize(fpath)
                all_files.append(entry)
        for entry in result_data.get('plots', []):
            fname = entry.get('filename', '')
            fpath = os.path.join(work_dir, fname)
            if os.path.exists(fpath):
                entry['size'] = os.path.getsize(fpath)
                all_files.append(entry)
        return all_files


sandbox_executor = SandboxExecutor()
