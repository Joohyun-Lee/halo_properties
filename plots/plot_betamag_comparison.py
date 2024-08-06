import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from halo_properties.utils.utils import gather_h5py_files
from halo_properties.utils.output_paths import gen_paths, dataset
from halo_properties.params.params import *
from halo_properties.utils.functions_latest import get_infos
from plot_functions.generic.stat import xy_stat
from plot_functions.generic.plot_functions import make_figure, xy_plot_stat
from plot_functions.UV.UVslope.betaVSmag import (
    make_magbeta,
    plot_magbeta,
    plot_magbeta_constraints,
    plot_dustier_magbeta,
)
from halo_properties.dust.att_coefs import get_dust_att_keys
import os
import h5py


def load_magbeta(sim_name, out_nb, dset, ext_key):
    out, assoc_out, analy_out, suffix = gen_paths(sim_name, out_nb, dset)

    # print(analy_out)

    keys = ["mag_" + ext_key, "betas_" + ext_key]

    # try:
    datas = gather_h5py_files(analy_out, keys=keys)
    # except OSError as e:
    # print(f'OSError : {out_nb:d} ll={ll:f}, {rtwo_fact:f}xr200, association: {assoc_mthd:s}, {fesc_key:s}, xkey={xkey:s}')
    # print(e)

    return [datas[k][()] for k in keys]


out_nb = 82
overwrite = False
# lls = [0.2, 0.2, 0.2, 0.2, 0.2]
# mps = [True, False, False, True, True]
# lls = [0.1, 0.15, 0.2, 0.2, 0.2]
assoc_mthds = [
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
    "stellar_peak",
    # "stellar_peak",
    # "stellar_peak",
]
# assoc_mthds = [
#     "stellar_peak",
#     "stellar_peak",
#     "stellar_peak",
#     "fof_ctr",
#     "stellar_peak",
# ]
# r200s = [1.0, 1.0, 1.0, 1.0, 1.0]
# rstars = [1.0, 1.0, 1.0, 1.0, 1.0]
# cleans = [True, True, False, True, True]
# max_dtms = [0.5, 0.5, 0.5, 0.1, 0.05]

lls = [0.2, 0.2, 0.2, 0.2]
assoc_mthds = ["stellar_peak", "stellar_peak", "stellar_peak", "stellar_peak"]
r200s = [1.0, 1.0, 1.0, 1.0]
rstars = [1.0, 1.0, 1.0, 1.0]
cleans = [True, True, True, True]
max_dtms = [0.05, 0.05, 0.10, 0.5]
mps = [True, True, False, False]
neb_cont_file_names = [
    None,
    "cstSFR_5Myr_nebular_continuum_SB99_pseudo-f=1_pf_width=200Angstrom_emissivites_mags_betas.txt",
    "cstSFR_5Myr_nebular_continuum_SB99_pseudo-f=1_pf_width=200Angstrom_emissivites_mags_betas.txt",
    None,
]
fig, ax = make_figure()
# ext_key = [k for k in get_dust_att_keys() if "WD_LMC2_10" in k][0]
ext_key = [k for k in get_dust_att_keys() if "LMCavg_20" in k][0]
nbins = 40
mag_bins = np.linspace(-25, -5, nbins)


# ext_key = [k for k in get_dust_att_keys() if "LMCavg_20" in k][0]


# nbins = 55
# mag_bins = np.linspace(-24, -10, nbins)

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

labels = []
lines = []

fig, ax = make_figure()

ncols = 6
nrows = 1


for iplot, (
    assoc_mthd,
    ll,
    r200,
    rstar,
    mp,
    clean,
    dtm_max,
    neb_cont_file_name,
) in enumerate(
    zip(assoc_mthds, lls, r200s, rstars, mps, cleans, max_dtms, neb_cont_file_names)
):
    dset = dataset(
        r200=r200,
        ll=ll,
        assoc_mthd=assoc_mthd,
        clean=clean,
        mp=mp,
        max_DTM=dtm_max,
        neb_cont_file_name=neb_cont_file_name,
    )

    out_path = os.path.join("./files")
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # overwrite = False
    # if r200 == 2:
    #     overwrite = True

    # overwrite = False
    # if dtm_max != 0.5:
    #     overwrite = True

    out_file = os.path.join(
        out_path,
        f"magbeta_{out_nb:d}_{assoc_mthd:s}_{ll:.2f}_{r200:.1f}_{ext_key:s}",
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
        mags, betas = load_magbeta("CoDaIII", out_nb, dset, ext_key)

        # print(mags.min(), mags.max(), np.mean(mags))

        bins, rel = make_magbeta(mags, betas, mag_bins)

        with h5py.File(out_file, "w") as dest:
            dest.create_dataset("mags", data=bins, dtype="f4")
            dest.create_dataset("magVbeta", data=rel, dtype="f4")

    else:
        with h5py.File(out_file, "r") as dest:
            bins = dest["mags"][()]
            rel = dest["magVbeta"][()]
    # elif overwrite and exists:

    #     with h5py.File(out_file, 'a') as dest:
    #         f_masses = dest["xbins"]
    #         f_fescs = dest["fescs"]

    #         f_masses[...] = xbins
    #         f_fescs[...] = ystat

    line = plot_magbeta(fig, ax, bins, rel, redshift=redshift, linewidth=3)

    lines.append(line)

    label = f"{assoc_mthd:s} ll={ll:.2f} {r200:.1f}Xr200 max(dtm) {dtm_max:.2f}"
    if rstar != 1.0:
        label += f" rstar={rstar:.1f}"
    if mp:
        label += " mp"
    if clean:
        label += " clean"
    if neb_cont_file_name is not None:
        label += f"w nebular"
    labels.append(label)


# lines = plot_fct(fig, ax, masses, fescs, fesc_type, redshift)
dustier_lines = []
dustier_labels = []

dustier_lines, dustier_labels = plot_dustier_magbeta(ax, redshift, bins, color="k")

labels += dustier_labels
lines += dustier_lines

obs_lines = []
obs_labels = []

obs_lines, obs_labels = plot_magbeta_constraints(ax, redshift, bins, delta_z=0.5)

labels += obs_labels
lines += obs_lines

ax.grid()

ax.set_ylim(-3.0, -1.0)
ax.set_xlim(-15, -23)

plt.legend(lines, labels, framealpha=0.0)

fig_name = f"./figs/magbeta_comparison_{out_nb:d}_{redshift:.1f}_{ext_key:s}.png"
fig.savefig(fig_name)
