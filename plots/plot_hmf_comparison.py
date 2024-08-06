import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.lines import Line2D

import os
import h5py

from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.utils.functions_latest import get_infos
from halo_properties.params.params import *

from plot_functions.generic.stat import mass_function
from plot_functions.generic.plot_functions import make_figure

# from plot_functions.smf.smfs import plot_constraints, smf_plot, make_smf
from plot_functions.generic.stat import mass_function
from plot_functions.generic.plot_functions import mf_plot
from scipy.stats import binned_statistic


def load_masses(sim_name, out_nb, dset):
    keys = ["Mst", "mass", "SFR10"]

    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(analy_out)

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return datas["mass"][()], datas["Mst"][()], datas["SFR10"][()]


Mp = 5.09e4

out_nb = 106
overwrite = False
lls = [0.2, 0.2, 0.2, 0.2]  # , 0.2]  # , 0.2]
mps = [True, False, True, False]  # , True]  # , True]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = [
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
]  # , "stellar_peak"]
# "stellar_peak",
# ]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
r200s = [1.0, 1.0, 1.0, 1.0]  # , 1.0]  # , 2.0]
rstars = [1.0, 1.0, 1.0, 1.0]  # , 1.0]  # , 1.0]
cleans = [True, True, False, False]  # , False]  # , True]
max_dtms = [0.5, 0.5, 0.5, 0.5]  # , 0.5]  # , 0.1]


nbins = 25
mass_bins = np.logspace(7.5, 12.5, nbins)

info_path = os.path.join(sim_path, "outputs", f"output_{out_nb:06d}", "group_000001")

(
    t,
    a,
    H0,
    om_m,
    om_l,
    om_k,
    om_b,
    unit_l,
    unit_d,
    unit_t,
    l,
    Lco,
    L,
    px_to_m,
) = get_infos(info_path, out_nb, ldx)


redshift = 1.0 / a - 1.0

assert (
    len(r200s) == len(assoc_mthds) == len(lls)
), "check input parameter lists' lengths"

masses = []
hmfs = []
fstars = []
fsfrs = []
labels = []
lines = []
errs = []
# lines = []

for iplot, (assoc_mthd, ll, r200, rstar, mp, clean, dtm_max) in enumerate(
    zip(assoc_mthds, lls, r200s, rstars, mps, cleans, max_dtms)
):
    dset = dataset(
        r200=r200, ll=ll, assoc_mthd=assoc_mthd, clean=clean, mp=mp, max_DTM=dtm_max
    )

    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # overwrite = False
    # if dtm_max != 0.5:
    #     overwrite = True

    out_file = os.path.join(
        out_path,
        f"hmfs_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}",
    )

    if dtm_max != 0.5:
        out_file += f"_dtm{dtm_max:.2f}"
    if mp:
        out_file += "_mp"
    if clean:
        out_file += "_clean"

    exists = os.path.isfile(out_file)

    # print(out_file, exists)

    if overwrite or not exists:
        halo_masses, stellar_masses, sfrs = load_masses("CoDaIII", out_nb, dset)

        bins, hmf, err = mass_function(halo_masses, mass_bins)

        stellar_occupancy = binned_statistic(
            halo_masses,
            stellar_masses > 0,
            bins=mass_bins,
            statistic="mean",
        )[0]

        quenched_fraction = binned_statistic(  # not really quenched but opposite but OK who cares that much...
            halo_masses,
            sfrs > 0,
            bins=mass_bins,
            statistic="mean",
        )[
            0
        ]

        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("xbins", data=bins, dtype="f4")
            dest.create_dataset("hmf", data=hmf, dtype="f4")
            dest.create_dataset("err", data=err, dtype="f4")
            dest.create_dataset("stellar_occupancy", data=stellar_occupancy, dtype="f4")
            dest.create_dataset("quenched_fraction", data=quenched_fraction, dtype="f4")

    else:
        with h5py.File(out_file, "r") as dest:
            bins = dest["xbins"][()]
            hmf = dest["hmf"][()]
            err = dest["err"][()]
            stellar_occupancy = dest["stellar_occupancy"][()]
            quenched_fraction = dest["quenched_fraction"][()]

    print(hmf)

    masses.append(bins)
    hmfs.append(hmf)
    errs.append(err)
    fstars.append(stellar_occupancy)
    fsfrs.append(quenched_fraction)
    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200"
    if rstar != 1.0:
        label += f" rstar={rstar:.1f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    labels.append(label)


# print(masses, smfs)
fig, axs = plt.subplots(
    2, 1, figsize=(8, 8), sharex=True, sharey=False, height_ratios=[3, 1]
)

# make top panel for using make_axes_locatable


for mass, hmf, err, fsfr, fstar in zip(hmfs, masses, errs, fsfrs, fstars):
    mf_line = mf_plot(
        fig,
        axs[0],
        hmf,
        mass,
        linestyle="-",
        xlabel=r"Halo Mass, $M_\odot$",
        yerrs=err,
    )

    fstar_line = axs[1].plot(hmf, fstar, linestyle="--", color=mf_line.get_color())

    fsfr_line = axs[1].plot(hmf, fsfr, linestyle=":", color=mf_line.get_color())
    line = tuple([mf_line] + fstar_line + fsfr_line)

    lines.append(line)

    # print(line)

axs[0].legend(lines, labels, framealpha=0.0)

axs[0].set_ylabel("HMF, $dn/d\log M_\odot$")

axs[1].legend(
    [Line2D([], [], c="k", ls="--"), Line2D([], [], c="k", ls=":")],
    ["Fraction of halos with stars", "Fraction of star-forming halos"],
    framealpha=0.0,
)

for ax in axs:
    ax.grid()

fig.savefig(f"./figs/hmf_comparison_{out_nb:d}", bbox_inches="tight")


axs[0].set_xlim(1e10, 5e12)
axs[0].set_ylim(1e7, 1e10)
fig.savefig(f"./figs/hmf_comparison_high_mass_{out_nb:d}", bbox_inches="tight")
