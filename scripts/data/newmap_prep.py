from __future__ import annotations
import os
from ..utils.audio_tools import read_and_save_aff_tester_file, build_aff_tester_arrays
from ..utils.config_utils import ROOT_DIR, PATHS
MAPTHIS_DIR = os.path.join(ROOT_DIR, PATHS.get('mapthis_dir', os.path.join('data', 'mapthis')))

def prepare_new_map(aff_path: str, output_name: str='mapthis', audio_name: str='base.ogg') -> str:
    """returns: str"""
    os.makedirs(MAPTHIS_DIR, exist_ok=True)
    if not os.path.isabs(aff_path):
        aff_path = os.path.abspath(os.path.join(ROOT_DIR, aff_path))
    out_path = os.path.join(MAPTHIS_DIR, f'{output_name}')
    read_and_save_aff_tester_file(aff_path, filename=out_path, audio_name=audio_name)
    return out_path + '.npz'

def build_new_map_arrays(aff_path: str, audio_name: str='base.ogg'):
    """returns: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]"""
    if not os.path.isabs(aff_path):
        aff_path = os.path.abspath(os.path.join(ROOT_DIR, aff_path))
    return build_aff_tester_arrays(aff_path, audio_name=audio_name)
if __name__ == '__main__':
    default_aff = os.path.join(ROOT_DIR, '3.aff')
    print('preparing mapthis.npz from:', default_aff)
    out = prepare_new_map(default_aff)
    print('saved:', out)
