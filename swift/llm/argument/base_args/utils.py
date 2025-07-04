# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union

######################################################################################
# 支持resume_from_checkpoint=True时，自动寻找最新的version下最新的checkpoint，并继续训练
######################################################################################
import re
from typing import Optional
from swift.utils import get_logger

logger = get_logger()


def is_versioned_dir(dirname: str) -> bool:
    """Check if directory follows vX-YYYYMMDD-HHMMSS format"""
    return bool(re.match(r'^v\d+-\d{8}-\d{6}$', dirname))


def find_latest_versioned_dir(base_dir: str) -> Optional[str]:
    """Find the latest versioned subdirectory under base_dir"""
    if not os.path.exists(base_dir):
        return None

    version_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
           and is_versioned_dir(d)
    ]
    if not version_dirs:
        return None

    # Sort by modification time (newest first)
    version_dirs.sort(
        key=lambda x: os.path.getmtime(os.path.join(base_dir, x)),
        reverse=True
    )
    return os.path.join(base_dir, version_dirs[0])


def find_latest_cp(path: bool = False, output_dir: str = None):
    from transformers.trainer_utils import get_last_checkpoint

    if not path:  # False case
        return None

    # Auto-resume logic
    if not output_dir:
        raise ValueError("Cannot auto-resume: `output_dir` is not set")
    try:
        # Case 1: If output_dir is already a versioned dir and no siblings exist
        output_dir = os.path.abspath(os.path.expanduser(output_dir))  # 将用户传入的路径统一处理为绝对路径​​
        dirname = os.path.basename(output_dir)
        parent_dir = os.path.dirname(output_dir)

        if is_versioned_dir(dirname):
            sibling_dirs = [
                d for d in os.listdir(parent_dir)
                if d != dirname and is_versioned_dir(d)
            ]
            if not sibling_dirs:
                # Directly search in this versioned dir
                latest_ckpt = get_last_checkpoint(output_dir)
                if latest_ckpt is None:
                    raise FileNotFoundError(f"No valid checkpoint found in {output_dir}")
                return latest_ckpt

        # Case 2: Find latest versioned dir under output_dir
        latest_version_dir = find_latest_versioned_dir(output_dir)
        if latest_version_dir is None:
            raise FileNotFoundError(f"No versioned subdirectory found in {output_dir}")

        latest_ckpt = get_last_checkpoint(latest_version_dir)
        if latest_ckpt is None:
            raise FileNotFoundError(f"No valid checkpoint found in {latest_version_dir}")
        return latest_ckpt
    except Exception as e:
        logger.warning(f"Failed to resume checkpoint: {str(e)}. Falling back to initial model.")
        return None  # 返回None会让训练从头开始


def to_abspath(path: Union[str, List[str], bool, None], check_path_exist: bool = False, output_dir: str = None) -> \
Union[str, List[str], None]:
    """Check the path for validity and convert it to an absolute path.

    Args:
        path: The path to be checked/converted
        check_path_exist: Whether to check if the path exists

    Returns:
        Absolute path
    """

    if path is None:
        return None
    elif isinstance(path, bool):
        return find_latest_cp(path, output_dir)
    elif isinstance(path, str):
        # Remove user path prefix and convert to absolute path.
        path = os.path.abspath(os.path.expanduser(path))
        if check_path_exist and not os.path.exists(path):
            raise FileNotFoundError(f"path: '{path}'")
        return path
    assert isinstance(path, list), f'path: {path}'
    res = []
    for v in path:
        res.append(to_abspath(v, check_path_exist))
    return res