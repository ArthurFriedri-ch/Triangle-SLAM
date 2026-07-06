# interface/slam_interface/io.py
import os
import glob
from .record import Record


def write(record: Record, out_dir: str, validate: bool = True) -> str:
    if validate:
        record.validate()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"kf_{record.index:06d}.npz")
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(record.to_bytes())
    os.replace(tmp, path)          # atomic: a reader never sees a half-written file
    return path


def read(path: str) -> Record:
    with open(path, "rb") as fh:
        return Record.from_bytes(fh.read())


def list_record_paths(in_dir: str):
    # zero-padded names sort in frame order lexicographically
    return sorted(glob.glob(os.path.join(in_dir, "kf_*.npz")))


def iter_records(in_dir: str):
    """The mapper's consumer loop, offline path."""
    for p in list_record_paths(in_dir):
        yield read(p)