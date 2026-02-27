#!/usr/bin/env python3
import os, subprocess, time, socket, json, sys, re
from pathlib import Path
from getpass import getpass
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import urllib.parse

PORT = 11435
REPO_DIR = Path("/home/engine/project")
BIN_PATH = REPO_DIR / "target" / "release" / "chorus"
LOG_PATH = REPO_DIR / "chorus.log"
CONFIG_PATH = Path.home() / ".config" / "chorus" / "config.toml"

def run(cmd, cwd=None, env=None):
    print(f">> {cmd}")
    subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=env)

def is_listening(host="127.0.0.1", port=PORT, timeout=1.5):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False

def wait_port(host="127.0.0.1", port=PORT, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if is_listening(host, port):
            return True
        time.sleep(1)
    return False

def make_request(url, data=None, method="GET"):
    """使用 urllib 发起 HTTP 请求"""
    try:
        if data:
            data = json.dumps(data).encode('utf-8')
            req = Request(url, data=data, headers={'Content-Type': 'application/json'}, method=method)
        else:
            req = Request(url, method=method)
        
        with urlopen(req, timeout=60) as response:
            content = response.read().decode('utf-8')
            return response.status, content
    except HTTPError as e:
        return e.code, e.read().decode('utf-8') if hasattr(e, 'read') else str(e)
    except URLError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)

# 0) 如果已经在跑，直接提示成功
if is_listening():
    print(f"Chorus 已在端口 {PORT} 运行 -> http://127.0.0.1:{PORT}")
else:
    # 1) 准备二进制：若不存在，则自动安装 Rust 并构建
    if not BIN_PATH.exists():
        print("未找到编译好的 Chorus，开始安装/构建...")
        run("apt-get update -y && apt-get install -y -q build-essential pkg-config libssl-dev git curl")
        run("curl https://sh.rustup.rs -sSf | sh -s -- -y")
        CARGO_BIN = str(Path.home() / ".cargo" / "bin")
        ENV = os.environ.copy()
        ENV["PATH"] = CARGO_BIN + ":" + ENV.get("PATH", "")

        run("cargo build --release -j 2", cwd=str(REPO_DIR), env=ENV)
    else:
        print("检测到已构建的 Chorus 二进制，跳过编译。")

    # 2) 准备最小可用配置（如不存在则创建）
    if not CONFIG_PATH.exists():
        print("未检测到配置文件，创建测试配置")
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        # 使用项目自带的测试配置
        test_cfg = Path(__file__).parent / "test-config.toml"
        if test_cfg.exists():
            cfg_content = test_cfg.read_text(encoding="utf-8")
            # 替换端口号为脚本指定的端口
            cfg_content = cfg_content.replace("port = 11435", f"port = {PORT}")
            CONFIG_PATH.write_text(cfg_content, encoding="utf-8")
            print(f"从 {test_cfg} 复制配置到：{CONFIG_PATH}")
        else:
            # 如果没有测试配置文件，创建基础配置
            cfg = f"""
[server]
host = "127.0.0.1"
port = {PORT}

[[model]]
name = "test-model"
api_base = "https://api.openai.com/v1"
api_key = "sk-test-key-for-validation"
auto_temperature = true

[workflow-integration]
nested_worker_depth = 1
json = \"\"\"
{{
  "analyzer": {{"ref": "test-model", "auto_temperature": true}},
  "workers": [{{"ref": "test-model"}}],
  "selector": {{"ref": "test-model"}},
  "synthesizer": {{"ref": "test-model"}}
}}
\"\"\"

[workflow]
[workflow.timeouts]
analyzer_timeout_secs = 30
worker_timeout_secs = 60
synthesizer_timeout_secs = 60

[workflow.domains]
"""
            CONFIG_PATH.write_text(cfg, encoding="utf-8")
            print(f"配置写入：{CONFIG_PATH}")
    else:
        print(f"使用已有配置：{CONFIG_PATH}")

    # 3) 启动服务（后台），写日志
    print("启动 Chorus 服务...")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    server_out = open(LOG_PATH, "wb")
    ENV2 = os.environ.copy()
    ENV2["RUST_LOG"] = "info"
    proc = subprocess.Popen([str(BIN_PATH)], cwd=str(REPO_DIR), env=ENV2,
                            stdout=server_out, stderr=subprocess.STDOUT)

    # 4) 等待端口就绪
    if not wait_port(timeout=180):
        server_out.close()
        print("\n启动失败：端口未就绪。最近 200 行日志如下：\n")
        try:
            print(subprocess.check_output(f"tail -n 200 {LOG_PATH}", shell=True, text=True))
        except Exception as e:
            print(f"(无法读取日志: {e})")
        raise SystemExit(1)

    print(f"Chorus 已就绪 -> http://127.0.0.1:{PORT}")

# 4.5) 使用 urllib 调用一次生成接口（替代原先的 curl，避免引号冲突）
try:
    prompt = """我是一名中国的16岁学习成绩差（语文、数学、英语以及物理都很差）的男开发者，在15岁时开发过一个调用多次llm，选一个最好的结果的工具，在15岁时尝试开发过一个能像大模型那样生成自然语言的内容的符号ai（名字是knowprolog），失败了，在15岁时开发了一个把typus（我和大模型一起创造的一个编程语言）编译到go代码的编译器，目前我不写代码，这两个的代码基本都是ai生成的。学过Rust和haskell（15岁时），没学完就放弃了。曾经开发过Chrome扩展（但是并不是特别会前端）（14岁左右时），也用AndroLua（最开始用的FusionApp）（同时搭配其他一些工具，如MT管理器）开发过安卓应用（9岁时用FusionApp开发过工具箱，写过lua代码）（10岁开始时用AndroLua开发过工具箱、相机和浏览器，写过lua代码），还用c语言开发过窗口管理器（13岁时）（好像过了3个月左右就放弃开发了）（但是并不是特别会c语言），走过一段时间oi路（12岁到13岁），后来放弃那条路了。这个人了解一些历史、地理、荣格八维（12岁开始接触）、mbti（12岁开始接触）、分子人类学（12岁开始大量接触）、linux（12岁开始大量接触）的知识，不过就一些，不是很多，同时知道一些手机的参数"""

    payload = {
        "model": "chorus",
        "prompt": prompt,
        "stream": False,
    }

    url = f"http://127.0.0.1:{PORT}/api/generate"
    status_code, response_text = make_request(url, data=payload, method="POST")
    print("生成接口状态码:", status_code)
    print("生成接口响应体前 1000 字符:\n", response_text[:1000] if response_text else "无响应")
except Exception as e:
    print("调用 /api/generate 失败：", e)

# 5) 快速健康检查
try:
    url = f"http://127.0.0.1:{PORT}/v1/models"
    status_code, response_text = make_request(url)
    if status_code == 200:
        print("健康检查 OK：/v1/models 可访问。")
    else:
        print("健康检查失败：/v1/models 返回非 200。状态码:", status_code, response_text[:300] if response_text else "无响应")
except Exception as e:
    print("健康检查异常：", e)
    print("\n可执行以下命令查看日志：")
    print(f"!tail -n 200 {LOG_PATH}")