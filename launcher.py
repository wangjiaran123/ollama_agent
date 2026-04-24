#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Agent 一体化自动化脚本
整合项目现有启动、测试、监控脚本为一个完整的自动化工作流。

功能:
  1. 环境检查（Python版本、依赖、文件完整性、Ollama服务）
  2. 全面测试流程（预检 → 启动应用 → 集成测试 → 清理）
  3. 失败时输出详细的诊断报告（错误类型、堆栈跟踪、环境快照）
  4. 测试通过后自动进入交互式启动菜单
  5. 完整的日志记录，全程可追踪、可调试

用法:
  python launcher.py             完整流程（检查 → 测试 → 启动）
  python launcher.py --check-only 仅运行环境检查
  python launcher.py --test-only  仅运行测试（不启动）
  python launcher.py --quick      跳过测试直接启动（不推荐）
"""

import sys
import os
import time
import json
import signal
import shutil
import subprocess
import textwrap
import importlib
import traceback
import platform
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"launcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

SANDBOX_DIR = PROJECT_ROOT / "sandbox_output"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

APP_HOST = "127.0.0.1"
APP_PORT = 5000
APP_URL = f"http://{APP_HOST}:{APP_PORT}"
OLLAMA_URL = "http://localhost:11434"

REQUIRED_FILES = [
    "app.py",
    "sandbox_executor.py",
    "static/index.html",
    "static/js/app.js",
    "static/js/socket.io.min.js",
    "static/css/style.css",
]

REQUIRED_PACKAGES = [
    ("flask", "flask"),
    ("flask_socketio", "flask-socketio"),
    ("requests", "requests"),
]

OPTIONAL_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("aiohttp", "aiohttp"),
    ("openai", "openai"),
]


class Logger:
    def __init__(self, log_file: Path):
        self.log_file = log_file
        self._buffer: List[str] = []
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, level: str, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] [{level}] {message}"
        self._buffer.append(line)
        print(message)

    def flush(self):
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(f"Ollama Agent Launcher Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Command: {' '.join(sys.argv)}\n")
                f.write(f"{'='*60}\n")
                for line in self._buffer:
                    f.write(line + "\n")
        except IOError as e:
            print(f"  [WARN] 无法写入日志文件 {self.log_file}: {e}")

    def info(self, message: str):
        self._write("INFO", message)

    def ok(self, message: str):
        self._write("OK", f"  [OK] {message}")

    def warn(self, message: str):
        self._write("WARN", f"  [WARN] {message}")

    def fail(self, message: str):
        self._write("FAIL", f"  [FAIL] {message}")

    def error(self, message: str):
        self._write("ERROR", f"  [ERROR] {message}")

    def section(self, title: str):
        line = f"\n{'='*60}"
        self._write("SECTION", f"{line}\n  {title}\n{line}")

    def divider(self):
        self._write("DIVIDER", "-" * 60)


log = Logger(LOG_FILE)


class EnvironmentSnapshot:
    @staticmethod
    def collect() -> Dict:
        info = {
            "timestamp": datetime.now().isoformat(),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
                "argv": sys.argv,
            },
            "project_root": str(PROJECT_ROOT),
            "cwd": os.getcwd(),
            "env_vars": {
                k: v for k, v in os.environ.items()
                if not k.startswith(" ") and not any(secret in k.lower() for secret in ["key", "token", "secret", "password", "credential"])
            },
            "installed_packages": [],
        }
        try:
            import pkg_resources
            info["installed_packages"] = sorted(
                [f"{d.key}=={d.version}" for d in pkg_resources.working_set]
            )
        except Exception:
            try:
                import importlib.metadata
                info["installed_packages"] = sorted(
                    [f"{dist.metadata['Name']}=={dist.version}"
                     for dist in importlib.metadata.distributions()]
                )
            except Exception:
                info["installed_packages"] = ["<unable to list packages>"]
        return info

    @staticmethod
    def format(info: Dict) -> str:
        lines = ["===== Environment Snapshot ====="]
        lines.append(f"Timestamp: {info['timestamp']}")
        lines.append(f"OS: {info['platform']['system']} {info['platform']['release']}")
        lines.append(f"Python: {info['python']['version'].strip()}")
        lines.append(f"Executable: {info['python']['executable']}")
        lines.append(f"Project Root: {info['project_root']}")
        lines.append(f"CWD: {info['cwd']}")
        lines.append("--- Installed Packages ---")
        for pkg in info["installed_packages"]:
            lines.append(f"  {pkg}")
        return "\n".join(lines)


class CheckReport:
    def __init__(self):
        self.checks: List[dict] = []
        self.start_time = time.time()

    def add(self, name: str, passed: bool, detail: str = ""):
        self.checks.append({"name": name, "passed": passed, "detail": detail})

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c["passed"])

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c["passed"])

    @property
    def all_passed(self) -> bool:
        return self.failed_count == 0

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def print_summary(self, title: str = ""):
        if title:
            log.section(title)
        for check in self.checks:
            if check["passed"]:
                log.ok(check["name"])
            else:
                log.fail(f"{check['name']} - {check['detail']}")
        log.divider()
        log.info(f"  通过: {self.passed_count}, 失败: {self.failed_count}, 耗时: {self.elapsed:.2f}s")
        log.divider()


def print_banner():
    banner = f"""
╔{'═'*58}╗
║                🤖 Ollama Agent - 一体化启动工具                ║
║                {'═'*38}                ║
║  工作目录: {str(PROJECT_ROOT):<44}║
║  日志文件: {str(LOG_FILE):<44}║
╚{'═'*58}╝
"""
    print(banner)
    log.info(f"启动 Ollama Agent 一体化脚本 (PID: {os.getpid()})")
    log.info(f"工作目录: {PROJECT_ROOT}")
    log.info(f"日志文件: {LOG_FILE}")


def check_environment() -> CheckReport:
    report = CheckReport()
    log.section("STEP 1/4: 环境检查")

    python_version = sys.version_info
    passed = python_version.major >= 3 and python_version.minor >= 8
    report.add(
        "Python 版本检查",
        passed,
        f"当前 {python_version.major}.{python_version.minor}.{python_version.micro}, 需要 >= 3.8"
    )

    for module_name, pkg_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(module_name)
            report.add(f"依赖包: {pkg_name}", True)
        except ImportError:
            report.add(f"依赖包: {pkg_name}", False, f"未安装，请运行: pip install {pkg_name}")

    for pkg_name in ["numpy", "pandas", "matplotlib", "PIL"]:
        try:
            importlib.import_module(pkg_name)
            report.add(f"可选包: {pkg_name}", True)
        except ImportError:
            report.add(f"可选包: {pkg_name}", True, "未安装（不影响核心功能）")

    for file_path in REQUIRED_FILES:
        full_path = PROJECT_ROOT / file_path
        exists = full_path.exists()
        report.add(
            f"文件: {file_path}",
            exists,
            "" if exists else f"缺失: {full_path}"
        )

    return report


def check_ollama_service() -> CheckReport:
    report = CheckReport()
    log.section("STEP 2/4: 外部服务检查")

    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            report.add("Ollama 服务连接", True, f"可用模型: {', '.join(model_names) if model_names else '无'}")
        else:
            report.add("Ollama 服务连接", True, f"响应状态码: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        report.add("Ollama 服务连接", True, "未运行（将使用外部API）")
    except Exception as e:
        report.add("Ollama 服务连接", True, f"检查异常: {e}")

    return report


class AppProcess:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None

    def start(self) -> bool:
        log.section("启动应用进程")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "app.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(PROJECT_ROOT),
            )
            self.start_time = time.time()
            log.info(f"应用进程已启动 (PID: {self.process.pid})")

            for attempt in range(15):
                time.sleep(1)
                if self._is_ready():
                    elapsed = time.time() - self.start_time
                    log.info(f"应用就绪，耗时 {elapsed:.1f}s")
                    return True
                if self.process.poll() is not None:
                    self._capture_failure_output()
                    return False
                if attempt % 5 == 4:
                    log.info(f"  等待应用启动... ({attempt + 1}s)")

            elapsed = time.time() - self.start_time
            log.warn(f"应用启动超时 ({elapsed:.1f}s)，进程可能仍在启动中")
            if self._is_ready():
                return True

            self.stop()
            return False

        except Exception as e:
            log.error(f"启动应用进程失败: {e}")
            log.error(traceback.format_exc())
            return False

    def _is_ready(self) -> bool:
        try:
            resp = requests.get(f"{APP_URL}/", timeout=2)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False
        except Exception:
            return False

    def _capture_failure_output(self):
        try:
            stdout, stderr = self.process.communicate(timeout=3)
            if stdout:
                log.error(f"应用标准输出:\n{textwrap.indent(stdout[-2000:], '    ')}")
            if stderr:
                log.error(f"应用错误输出:\n{textwrap.indent(stderr[-2000:], '    ')}")
        except Exception:
            pass

    def stop(self):
        if self.process is None:
            return
        log.info("正在停止应用进程...")
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
            log.info("应用进程已正常终止")
        except subprocess.TimeoutExpired:
            log.warn("应用进程未响应终止信号，强制终止...")
            try:
                self.process.kill()
                self.process.wait(timeout=3)
                log.info("应用进程已强制终止")
            except Exception as e:
                log.error(f"强制终止应用进程失败: {e}")
        except Exception as e:
            log.error(f"停止应用进程异常: {e}")
        finally:
            self.process = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class TestRunner:
    def __init__(self, app: AppProcess):
        self.app = app

    def run_all(self) -> CheckReport:
        report = CheckReport()
        log.section("STEP 3/4: 运行集成测试")

        self._test_endpoint(report, "主页", "/", lambda r: r.status_code == 200)
        self._test_endpoint(report, "API: 模型列表", "/api/models",
                            lambda r: r.json().get("success") is True)
        self._test_endpoint(report, "API: 系统配置", "/api/config",
                            lambda r: "external_api_base" in r.json())
        self._test_endpoint(report, "API: 文件上传", "/api/upload",
                            lambda r: r.json().get("success") is True,
                            method="POST",
                            files={"file": ("test.txt", b"hello world", "text/plain")})
        self._test_endpoint(report, "API: Ollama 连接检查", "/api/test-ollama",
                            lambda r: r.json().get("success") is True,
                            method="POST")

        self._test_code_execution(report)

        return report

    def _test_endpoint(self, report: CheckReport, name: str, path: str,
                       validator, method: str = "GET", **kwargs):
        try:
            url = f"{APP_URL}{path}"
            if method == "GET":
                resp = requests.get(url, timeout=10)
            else:
                resp = requests.post(url, timeout=15, **kwargs)
            passed = validator(resp)
            detail = f"HTTP {resp.status_code}" if not passed else ""
            report.add(name, passed, detail)
        except requests.exceptions.ConnectionError:
            report.add(name, False, "连接被拒绝")
        except requests.exceptions.Timeout:
            report.add(name, False, "请求超时")
        except Exception as e:
            report.add(name, False, str(e))

    def _test_code_execution(self, report: CheckReport):
        basic_result = self._execute_code("print('hello from sandbox')")
        report.add(
            "代码解释器: 基本执行",
            basic_result.get("success") is True,
            f"output={basic_result.get('stdout', '')[:50]}" if basic_result.get("success") else f"error={basic_result.get('error', '')[:100]}"
        )

        blocked_result = self._execute_code("import os\nprint(os.name)")
        blocked = blocked_result.get("success") is False
        report.add(
            "代码解释器: 安全沙盒拦截",
            blocked,
            "危险导入被正确拦截" if blocked else f"安全检查失效: {blocked_result.get('error', '')[:100]}"
        )

        numpy_result = self._execute_code("import numpy as np\nprint(np.array([1,2,3]).sum())")
        report.add(
            "代码解释器: NumPy支持",
            numpy_result.get("success") is True,
            f"output={numpy_result.get('stdout', '')[:30]}" if numpy_result.get("success") else f"error={numpy_result.get('error', '')[:100]}"
        )

        plot_result = self._execute_code(
            "import matplotlib.pyplot as plt\nimport numpy as np\n"
            "plt.plot([1,2,3])\nplt.show()"
        )
        has_plot = len(plot_result.get("plots", [])) > 0
        report.add(
            "代码解释器: 图表生成",
            plot_result.get("success") is True and has_plot,
            f"图表数: {len(plot_result.get('plots', []))}" if has_plot else f"error={plot_result.get('error', '')[:100]}"
        )

    def _execute_code(self, code: str) -> dict:
        try:
            resp = requests.post(
                f"{APP_URL}/api/code/execute",
                json={"code": code, "session_id": "launcher_test", "timeout": 30, "tags": ["cache"]},
                timeout=35,
            )
            data = resp.json()
            if data.get("success") and data.get("result"):
                return data["result"]
            return {"success": False, "error": str(data)}
        except Exception as e:
            return {"success": False, "error": str(e)}


def collect_failure_diagnostics(test_report: CheckReport) -> str:
    log.section("收集失败诊断信息")

    snapshot = EnvironmentSnapshot.collect()
    env_text = EnvironmentSnapshot.format(snapshot)

    diagnostic = [
        "=" * 60,
        "  FAILURE DIAGNOSTIC REPORT",
        "=" * 60,
        f"Generated: {datetime.now().isoformat()}",
        "",
        "--- Test Results ---",
        f"Passed: {test_report.passed_count}/{len(test_report.checks)}",
        f"Failed: {test_report.failed_count}/{len(test_report.checks)}",
        "",
        "--- Failed Checks ---",
    ]

    for check in test_report.checks:
        if not check["passed"]:
            diagnostic.append(f"  - {check['name']}")
            if check["detail"]:
                diagnostic.append(f"    Detail: {check['detail']}")

    diagnostic.extend([
        "",
        "--- Failure Summary ---",
        f"Total checks: {len(test_report.checks)}",
        f"Total passed: {test_report.passed_count}",
        f"Total failed: {test_report.failed_count}",
        f"Elapsed: {test_report.elapsed:.2f}s",
        "",
        "--- Environment Snapshot ---",
        env_text,
        "",
        "=" * 60,
        "  END OF DIAGNOSTIC REPORT",
        "=" * 60,
    ])

    return "\n".join(diagnostic)


def cleanup_cache():
    log.section("清理缓存和冗余文件")

    cleaned_dirs = 0
    cleaned_files = 0
    freed_bytes = 0

    sandbox_path = str(SANDBOX_DIR)
    log.info(f"扫描 sandbox_output 目录: {sandbox_path}")

    if SANDBOX_DIR.exists():
        for item in list(SANDBOX_DIR.iterdir()):
            if item.is_dir() and item.name.startswith("run_"):
                cache_tag = item / ".cache_tag"
                if not cache_tag.exists():
                    log.info(f"跳过非测试目录: {item.name} (用户创建的代码保留)")
                    continue
                try:
                    size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                    shutil.rmtree(item)
                    cleaned_dirs += 1
                    freed_bytes += size
                    log.ok(f"已删除: {item.name} (释放 {_format_size(size)})")
                except Exception as e:
                    log.error(f"删除 {item.name} 失败: {e}")
    else:
        log.info("sandbox_output 目录不存在，跳过")

    log.info("扫描 Python 缓存文件...")
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if ".venv" in pycache.parts or ".git" in pycache.parts:
            continue
        try:
            size = sum(f.stat().st_size for f in pycache.rglob("*") if f.is_file())
            shutil.rmtree(pycache)
            cleaned_dirs += 1
            freed_bytes += size
            log.ok(f"已删除缓存目录: {pycache.relative_to(PROJECT_ROOT)}")
        except Exception as e:
            log.error(f"删除 {pycache} 失败: {e}")

    for pyc_file in PROJECT_ROOT.rglob("*.pyc"):
        if ".venv" in pyc_file.parts or ".git" in pyc_file.parts:
            continue
        if pyc_file.parent.name == "__pycache__":
            continue
        try:
            size = pyc_file.stat().st_size
            pyc_file.unlink()
            cleaned_files += 1
            freed_bytes += size
        except Exception as e:
            log.error(f"删除 {pyc_file} 失败: {e}")

    log.info("保留 logs 目录中的异常日志文件")
    log.info("保留 uploads 目录中的用户文件")

    log.divider()
    total_freed = _format_size(freed_bytes)
    log.info(f"清理完成: 删除 {cleaned_dirs} 个目录, {cleaned_files} 个文件")
    log.info(f"释放空间: {total_freed}")
    log.divider()

    print(f"\n  ✅ 清理完成!")
    print(f"  删除: {cleaned_dirs} 个目录, {cleaned_files} 个文件")
    print(f"  释放: {total_freed}")
    print(f"  异常日志: 已保留\n")


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


def show_interactive_menu() -> Optional[str]:
    log.section("启动选择")

    menu = textwrap.dedent("""\
    ╔══════════════════════════════════════╗
    ║        Ollama Agent 启动选项         ║
    ╠══════════════════════════════════════╣
    ║  1. 🌐  仅启动 Web 应用 (推荐)       ║
    ║  2. 📊  同时启动监控工具              ║
    ║  3. 📈  仅启动监控工具                ║
    ║  4. 🧹  清理缓存和冗余文件            ║
    ║  5. 🚪  退出                          ║
    ╚══════════════════════════════════════╝
    """)
    print(menu)

    while True:
        try:
            choice = input("  请输入选择 (1-5): ").strip()
            if choice in ("1", "2", "3", "4", "5"):
                return choice
            print("  [WARN] 无效选择，请输入 1-5")
        except (EOFError, KeyboardInterrupt):
            print()
            return "5"


def launch_web_app():
    log.info("启动 Web 应用...")
    print(f"\n  🌐 访问地址: http://localhost:{APP_PORT}")
    print(f"  ⏹️  按 Ctrl+C 停止服务\n")
    try:
        subprocess.run([sys.executable, "app.py"], cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n  应用已停止")
    except Exception as e:
        log.error(f"启动 Web 应用失败: {e}")
        log.error(traceback.format_exc())
        print(f"\n  [ERROR] 启动失败: {e}")


def launch_monitor():
    log.info("启动命令行监控...")
    print(f"\n  📊 监控工具启动中...\n  ⏹️  按 'q' 键退出\n")
    try:
        subprocess.run([sys.executable, "monitor.py"], cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n  监控已停止")
    except Exception as e:
        log.error(f"启动监控失败: {e}")
        log.error(traceback.format_exc())
        print(f"\n  [ERROR] 启动失败: {e}")


def handle_menu_choice(choice: str):
    choice_map = {
        "1": ("🌐  仅启动 Web 应用", lambda: launch_web_app()),
        "2": ("📊  启动 Web + 监控", lambda: _launch_both()),
        "3": ("📈  仅启动监控工具", lambda: launch_monitor()),
        "4": ("🧹  清理缓存和冗余文件", lambda: cleanup_cache()),
        "5": ("🚪  退出", lambda: None),
    }
    desc, action = choice_map[choice]
    log.info(f"用户选择: {desc}")
    print(f"\n  您选择了: {desc}\n")
    action()


def _launch_both():
    log.info("同时启动 Web 应用和监控...")
    monitor_proc = None
    try:
        monitor_proc = subprocess.Popen(
            [sys.executable, "monitor.py"],
            cwd=str(PROJECT_ROOT),
        )
        time.sleep(2)
        if monitor_proc.poll() is not None:
            log.warn("监控工具未能正常启动")
        launch_web_app()
    except Exception as e:
        log.error(f"启动失败: {e}")
    finally:
        if monitor_proc is not None:
            try:
                monitor_proc.terminate()
                monitor_proc.wait(timeout=3)
            except Exception:
                pass


def run_pipeline(check_only: bool = False, test_only: bool = False, quick: bool = False) -> int:
    print_banner()

    env_report = check_environment()
    env_report.print_summary()

    if not env_report.all_passed:
        log.error("环境检查失败，终止执行")
        log.flush()
        return 1

    ollama_report = check_ollama_service()
    ollama_report.print_summary()

    if check_only:
        log.info("--check-only 模式: 环境检查完成，跳过测试和启动")
        log.flush()
        return 0

    if quick:
        log.warn("--quick 模式: 跳过测试，直接进入启动菜单")
        log.flush()
        while True:
            choice = show_interactive_menu()
            if choice == "5":
                log.info("用户选择退出")
                break
            handle_menu_choice(choice)
        log.flush()
        return 0

    app_proc = AppProcess()
    if not app_proc.start():
        log.error("应用启动失败，终止测试流程")
        log.flush()
        return 1

    test_runner = TestRunner(app_proc)
    test_report = test_runner.run_all()
    test_report.print_summary("集成测试结果")

    app_proc.stop()

    if test_report.all_passed:
        log.info("所有测试通过！")
        log.divider()

        if test_only:
            log.info("--test-only 模式: 测试完成，跳过启动")
            log.flush()
            return 0

        while True:
            choice = show_interactive_menu()
            if choice == "5":
                log.info("用户选择退出")
                log.flush()
                return 0
            handle_menu_choice(choice)
    else:
        log.error("测试未全部通过")

        diagnostic = collect_failure_diagnostics(test_report)

        report_file = LOG_DIR / f"failure_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(diagnostic)
            log.info(f"诊断报告已保存: {report_file}")
        except IOError as e:
            log.error(f"无法写入诊断报告: {e}")

        log.info(f"\n{'='*60}")
        log.info(f"  测试失败: {test_report.failed_count} 项未通过")
        log.info(f"  详细诊断报告: {report_file}")
        log.info(f"  修复问题后重新运行: python launcher.py")

        log.flush()
        return 1


def main():
    args = set(sys.argv[1:])

    check_only = "--check-only" in args or "-c" in args
    test_only = "--test-only" in args or "-t" in args
    quick = "--quick" in args or "-q" in args
    show_help = "--help" in args or "-h" in args

    if show_help:
        print(__doc__)
        return 0

    try:
        exit_code = run_pipeline(check_only, test_only, quick)
    except KeyboardInterrupt:
        print("\n\n  用户中断执行")
        log.info("用户中断执行")
        log.flush()
        exit_code = 130
    except Exception as e:
        print(f"\n  [ERROR] 未预期的错误: {e}")
        print(traceback.format_exc())
        log.error(f"未预期的错误: {e}")
        log.error(traceback.format_exc())
        log.flush()
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
