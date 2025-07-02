# Copyright (c) Alibaba, Inc. and its affiliates.
import importlib.util
import os
import subprocess
import sys
from typing import Dict, List, Optional

from swift.utils import get_logger

logger = get_logger()

ROUTE_MAPPING: Dict[str, str] = {
    'pt': 'swift.cli.pt',
    'sft': 'swift.cli.sft',
    'infer': 'swift.cli.infer',
    'merge-lora': 'swift.cli.merge_lora',
    'web-ui': 'swift.cli.web_ui',
    'deploy': 'swift.cli.deploy',
    'rollout': 'swift.cli.rollout',
    'rlhf': 'swift.cli.rlhf',
    'sample': 'swift.cli.sample',
    'export': 'swift.cli.export',
    'eval': 'swift.cli.eval',
    'app': 'swift.cli.app',
}


def use_torchrun() -> bool:
    nproc_per_node = os.getenv('NPROC_PER_NODE')
    nnodes = os.getenv('NNODES')
    if nproc_per_node is None and nnodes is None:
        return False
    return True


def get_torchrun_args() -> Optional[List[str]]:
    if not use_torchrun():
        return
    torchrun_args = []
    for env_key in ['NPROC_PER_NODE', 'MASTER_PORT', 'NNODES', 'NODE_RANK', 'MASTER_ADDR']:
        env_val = os.getenv(env_key)
        if env_val is None:
            continue
        torchrun_args += [f'--{env_key.lower()}', env_val]
    return torchrun_args


def _compat_web_ui(argv):
    # [compat]
    method_name = argv[0]
    if method_name in {'web-ui', 'web_ui'} and ('--model' in argv or '--adapters' in argv or '--ckpt_dir' in argv):
        argv[0] = 'app'
        logger.warning('Please use `swift app`.')


def cli_main0(route_mapping: Optional[Dict[str, str]] = None) -> None:
    sys.argv = [
        "swift/cli/main.py", 'sft',
        '--model', r'D:\projects\models\qwen3\Qwen3-0.6B',
        '--train_type', 'full',  # 关键修改：从'lora'改为'full'
        '--dataset', r'D:\projects\datasets\self-cognition#16',  # 数据量减少10倍（CPU调试用）
        r'D:\projects\datasets\Qwen3-SFT-Mixin#16',  # 数据量减少10倍
        '--torch_dtype', 'float32',  # CPU不支持bfloat16
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1',
        '--learning_rate', '1e-5',  # 全参数学习率需更低
        '--gradient_accumulation_steps', '4',  # 大幅减少（原16）
        '--eval_steps', '2',  # 更频繁验证（数据量少）
        '--save_steps', '2',
        '--save_total_limit', '1',
        '--logging_steps', '5',
        '--max_length', '512',  # 缩短序列长度（降低内存）
        '--output_dir', 'output_cpu',
        '--warmup_ratio', '0.05',
        '--dataloader_num_workers', '1',  # 减少workers（CPU限制）
        '--model_author', 'swift',
        '--model_name', 'swift-robot',
        '--no_cuda',  # 强制使用CPU
        '--gradient_checkpointing',  # 大幅减少内存占用
        '--optim', 'adamw_torch',  # 明确指定CPU友好优化器
        '--max_grad_norm', '1.0'  # 梯度裁剪（稳定训练）
    ]
    route_mapping = route_mapping or ROUTE_MAPPING
    argv = sys.argv[1:]
    _compat_web_ui(argv)
    method_name = argv[0].replace('_', '-')  # 'sft'
    argv = argv[1:]  # 原始参数去掉'sft'
    file_path = importlib.util.find_spec(
        route_mapping[method_name]).origin  # 'swift/cli/sft.py'  # 通过ROUTE_MAPPING['sft']找到模块
    torchrun_args = get_torchrun_args()
    python_cmd = sys.executable  # 当前Python解释器路径D:\projects\github\ms-swift\.venv\Scripts\python.exe
    if torchrun_args is None or method_name not in {'pt', 'sft', 'rlhf', 'infer'}:
        args = [python_cmd, file_path, *argv]
    else:
        args = [python_cmd, '-m', 'torch.distributed.run', *torchrun_args, file_path, *argv]
    print(f"run sh: `{' '.join(args)}`", flush=True)
    result = subprocess.run(args)
    if result.returncode != 0:
        sys.exit(result.returncode)


def cli_main(route_mapping: Optional[Dict[str, str]] = None) -> None:
    sys.argv = [
        r'swift/llm/train/sft.py',
        '--model', r'D:\projects\models\qwen3\Qwen3-0.6B',
        '--train_type', 'full',  # 关键修改：从'lora'改为'full'
        '--dataset', r'D:\projects\datasets\self-cognition#16',  # 数据量减少10倍（CPU调试用）
        r'D:\projects\datasets\Qwen3-SFT-Mixin#16',  # 数据量减少10倍
        '--torch_dtype', 'float32',  # CPU不支持bfloat16
        '--num_train_epochs', '1',
        '--per_device_train_batch_size', '1',
        '--per_device_eval_batch_size', '1',
        '--learning_rate', '1e-5',  # 全参数学习率需更低
        '--gradient_accumulation_steps', '4',  # 大幅减少（原16）
        '--eval_steps', '2',  # 更频繁验证（数据量少）
        '--save_steps', '2',
        '--save_total_limit', '1',
        '--logging_steps', '5',
        '--max_length', '512',  # 缩短序列长度（降低内存）
        '--output_dir', 'output_cpu',
        '--warmup_ratio', '0.05',
        '--dataloader_num_workers', '1',  # 减少workers（CPU限制）
        '--model_author', 'swift',
        '--model_name', 'swift-robot',
        '--no_cuda',  # 强制使用CPU
        '--gradient_checkpointing',  # 大幅减少内存占用
        '--optim', 'adamw_torch',  # 明确指定CPU友好优化器
        '--max_grad_norm', '1.0'  # 梯度裁剪（稳定训练）
    ]
    from swift.llm import sft_main
    sft_main()


if __name__ == '__main__':
    cli_main()