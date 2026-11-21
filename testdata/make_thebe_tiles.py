"""Cut ``thebe_tiles.npz`` from the Thebe fault-recognition dataset.

Source: An, Y., et al., 2021, A gigabyte interpreted seismic dataset for automatic
fault recognition: Data in Brief, 37, 107219. Data: https://doi.org/10.7910/DVN/YBYGBK (Harvard Dataverse, CC0).

The tiles come from the first training volume, ``data/npzfiles/seis/seistrain1.npz``
and its labels ``data/npzfiles/fault/faulttrain1.npz``, both of shape
(100, 3174, 1537): (section, trace, time sample). Each tile is 200 time samples by
256 traces, normalised to unit standard deviation and stored as float16; the fault
labels are bit-packed.

Usage::

    python make_thebe_tiles.py seistrain1.npz faulttrain1.npz thebe_tiles.npz
"""

from __future__ import annotations

import sys
import zipfile

import numpy as np
import numpy.typing as npt

# (section, first time sample, first trace)
TILES = [(95, 950, 1216), (25, 900, 512), (45, 800, 448), (75, 1150, 192)]
HEIGHT, WIDTH = 200, 256


def stream_tiles(path: str) -> npt.NDArray[np.float64]:
    """Read the tiles from the single array in an ``.npz`` file, one section at a
    time, without loading the whole volume."""
    sections = {s for s, _, _ in TILES}
    out = np.zeros((len(TILES), HEIGHT, WIDTH))
    with zipfile.ZipFile(path) as archive, archive.open(archive.infolist()[0]) as fh:
        version = np.lib.format.read_magic(fh)
        read_header = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, fortran, dtype = read_header(fh)
        if fortran or dtype.hasobject:
            msg = "expected a C-ordered numeric array"
            raise ValueError(msg)
        section_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        for section in range(shape[0]):
            raw = fh.read(section_bytes)
            if section not in sections:
                continue
            data = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
            for n, (s, t0, x0) in enumerate(TILES):
                if s == section:
                    out[n] = data[x0 : x0 + WIDTH, t0 : t0 + HEIGHT].T
    return out


def main() -> None:
    seis_path, fault_path, out_path = sys.argv[1:4]
    seismic = stream_tiles(seis_path)
    seismic /= seismic.std(axis=(1, 2), keepdims=True)
    faults = stream_tiles(fault_path) > 0.5
    np.savez_compressed(
        out_path,
        seismic=seismic.astype(np.float16),
        faults=np.packbits(faults, axis=-1),
        tiles=np.array(TILES),
        source=np.array(
            "Thebe fault dataset, An et al. (2021), doi:10.7910/DVN/YBYGBK, CC0"
        ),
    )


if __name__ == "__main__":
    main()
