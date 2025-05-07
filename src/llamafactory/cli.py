# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import sys
import random
from copy import deepcopy
from functools import partial


USAGE = (
    "-" * 70
    + "\n"
    + "| Usage:                                                             |\n"
    + "|   llamafactory-cli api -h: launch an OpenAI-style API server       |\n"
    + "|   llamafactory-cli chat -h: launch a chat interface in CLI         |\n"
    + "|   llamafactory-cli eval -h: evaluate models                        |\n"
    + "|   llamafactory-cli export -h: merge LoRA adapters and export model |\n"
    + "|   llamafactory-cli train -h: train models                          |\n"
    + "|   llamafactory-cli webchat -h: launch a chat interface in Web UI   |\n"
    + "|   llamafactory-cli webui: launch LlamaBoard                        |\n"
    + "|   llamafactory-cli version: show version info                      |\n"
    + "-" * 70
)


def main():
    from . import launcher
    from .api.app import run_api
    from .chat.chat_model import run_chat
    from .eval.evaluator import run_eval
    from .extras import logging
    from .extras.env import VERSION, print_env
    from .extras.misc import find_available_port, get_device_count, is_env_enabled, use_ray
    from .train.tuner import export_model, run_exp
    from .webui.interface import run_web_demo, run_web_ui

    logger = logging.get_logger(__name__)

    WELCOME = (
        "-" * 58
        + "\n"
        + f"| Welcome to LLaMA Factory, version {VERSION}"
        + " " * (21 - len(VERSION))
        + "|\n|"
        + " " * 56
        + "|\n"
        + "| Project page: https://github.com/hiyouga/LLaMA-Factory |\n"
        + "-" * 58
    )

    COMMAND_MAP = {
        "api": run_api,
        "chat": run_chat,
        "env": print_env,
        "eval": run_eval,
        "export": export_model,
        "train": run_exp,
        "webchat": run_web_demo,
        "webui": run_web_ui,
        "version": partial(print, WELCOME),
        "help": partial(print, USAGE),
    }

    command = sys.argv.pop(1) if len(sys.argv) >= 1 else "help"
    if command == "train" and (is_env_enabled("FORCE_TORCHRUN") or (get_device_count() > 1 and not use_ray())):
        force_torchrun = os.environ.get("FORCE_TORCHRUN", "0").lower() in ["true", "1"]
        if force_torchrun or get_device_count() > 1:
            master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
            master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
            logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
            process = subprocess.run(
                (
                    "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                    "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
                ).format(
                    nnodes=os.environ.get("NNODES", "1"),
                    node_rank=os.environ.get("RANK", "0"),
                    nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
                    master_addr=master_addr,
                    master_port=master_port,
                    file_name=launcher.__file__,
                    args=" ".join(sys.argv[1:]),
                ),
                shell=True,
            )
            sys.exit(process.returncode)
        else:
            run_exp()
    elif command in COMMAND_MAP:
        COMMAND_MAP[command]()
    else:
        print(f"Unknown command: {command}.\n{USAGE}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
