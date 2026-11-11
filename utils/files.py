"""File-system helpers shared by both analysis modes."""
import os
import re
from typing import List, Optional, Tuple

import config


def natural_sort_key(s: str):
    """Sort key so that 'img2' < 'img10'."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def find_channel_folders(root_folder: str) -> Optional[Tuple[str, str]]:
    """Return ``(fluorescence_dir, brightfield_dir)`` for the first valid pair, or None."""
    for fluo_name, bright_name in config.CHANNEL_FOLDER_PAIRS:
        fluo = os.path.join(root_folder, fluo_name)
        bright = os.path.join(root_folder, bright_name)
        if os.path.isdir(fluo) and os.path.isdir(bright):
            return fluo, bright
    return None


def list_images(folder: str) -> List[str]:
    """List image file names in ``folder``, naturally sorted."""
    files = [f for f in os.listdir(folder) if f.lower().endswith(config.IMAGE_EXTENSIONS)]
    return sorted(files, key=natural_sort_key)
