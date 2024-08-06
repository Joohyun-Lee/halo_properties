# Ouvrir un fullbox
import numpy as np
import os
import h5py
import sys


def o_data(data_pth):
    """
    Fetch data from fortran binary at data_pth
    """

    buf_size = 2147483639
    with open(data_pth, "rb", buf_size) as ff:
        bal = np.fromfile(ff, dtype=np.int32, count=1)[0]
        nx = np.fromfile(ff, dtype=np.int32, count=1)[0]
        ny = np.fromfile(ff, dtype=np.int32, count=1)[0]
        nz = np.fromfile(ff, dtype=np.int32, count=1)[0]
        bal = np.fromfile(ff, dtype=np.int32, count=1)[0]

        bitdata = np.zeros((nx * ny * nz * 4), dtype="S1")

        counter = 0
        while counter < (nx * ny * nz) * 4:
            bal = np.fromfile(ff, dtype=np.int32, count=1)[0]
            abal = abs(bal)
            bitdata[counter : counter + abal] = np.frombuffer(ff.read(abal), dtype="S1")
            counter += abal
            bal2 = np.fromfile(ff, dtype=np.int32, count=1)[0]

    return np.reshape(np.frombuffer(bitdata, dtype="f"), (nz, ny, nx), order="A")


def o_data_memmap(data_pth, slices=None):
    """
    Fetch data from fortran binary at data_pth
    """

    buf_size = 2147483639

    with open(data_pth, "rb", buf_size) as ff:
        bal = np.fromfile(ff, dtype=np.int32, count=1)[0]
        nx = np.fromfile(ff, dtype=np.int32, count=1)[0]
        ny = np.fromfile(ff, dtype=np.int32, count=1)[0]
        nz = np.fromfile(ff, dtype=np.int32, count=1)[0]
        bal = np.fromfile(ff, dtype=np.int32, count=1)[0]

    init_offset = 6 * 4
    tot_size = nx * ny * nz

    if tot_size * 4 > buf_size:
        raise Exception("multiple fortran buffer not supported")
    else:
        # get_size = tot_size
        pass

    # bitdata = np.zeros((nx * ny * nz * 4), dtype="S1")

    if slices == None:
        x0, y0, z0 = 0, 0, 0
        x1, y1, z1 = nx, ny, nz
    else:
        ((x0, x1), (y0, y1), (z0, z1)) = slices

    # print(x0, x1, y0, y1, z0, z1, nx, ny, nz)

    return np.squeeze(
        np.memmap(
            data_pth, offset=init_offset, shape=(nx, ny, nz), dtype="f4", mode="r"
        )[x0:x1, y0:y1, z0:z1]
    )


def o_data_hdf5(data_pth, key, slices):
    """
    Fetch data from fortran binary at data_pth
    """

    ((x0, x1), (y0, y1), (z0, z1)) = slices

    with h5py.File(data_pth, "r") as src:
        return src[key][x0:x1, y0:y1, z0:z1]
