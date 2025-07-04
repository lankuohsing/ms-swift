# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import List, Union


def to_abspath(path: Union[str, List[str], bool, None],  check_path_exist: bool = False, output_dir: str = None) -> Union[str, List[str], None]:
    """Check the path for validity and convert it to an absolute path.

    Args:
        path: The path to be checked/converted
        check_path_exist: Whether to check if the path exists

    Returns:
        Absolute path
    """
    if path is None:
        return
    elif isinstance(path, bool):
        if path is True:
            # 自动查找 output_dir 下的最新 checkpoint
            if not output_dir:
                raise ValueError("Cannot auto-resume: `output_dir` is not set")
            from transformers.trainer_utils import get_last_checkpoint
            latest_ckpt = get_last_checkpoint(output_dir)
            if latest_ckpt is None:
                raise FileNotFoundError(f"No valid checkpoint found in {output_dir}")
            return latest_ckpt
        else:
            return None
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
