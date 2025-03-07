"""
Plotting utilities for outputs of `SetLimit`
"""
import glob
import logging
from math import erf

import h5py
import legendstyles
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from legendfreqfit.models.constants import (
    NME_PHENO_HIGH,
    NME_PHENO_LOW,
    NME_central,
    NME_unc_hi,
    NME_unc_lo,
    m_prime_to_m,
    s_prime_to_s,
)
from legendfreqfit.statistics import (
    find_crossing,
    get_p_values,
    sensitivity,
    ts_critical,
)

plt.style.use(legendstyles.LEGEND)

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator


# define an object that will be used by the legend
class MulticolorPatch:
    def __init__(self, colors):
        self.colors = colors


# define a handler for the MulticolorPatch object
class MulticolorPatchHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        width, height = handlebox.width, handlebox.height
        patches = []
        for i, c in enumerate(orig_handle.colors):
            patches.append(
                plt.Rectangle(
                    [
                        width / len(orig_handle.colors) * i - handlebox.xdescent,
                        -handlebox.ydescent,
                    ],
                    width / len(orig_handle.colors),
                    height,
                    facecolor=c,
                    edgecolor="none",
                )
            )

        patch = PatchCollection(patches, match_original=True)

        handlebox.add_artist(patch)
        return patch


PLTSIZE = legendstyles.figsizes.JupyterNotebook

GOOD_ORANGE = "#ff5d07"
GOOD_MAGENTA = "#D907FF"
NICE_BLUE = "#668DA5"
NICE_RED = "#B4584D"
NICE_GREEN = "#ABB1A2"
NICE_PINK = "#CCACAD"


BRAZIL_1_SIGMA_KWARGS = {"color": "#07A9FF", "alpha": 1}
BRAZIL_2_SIGMA_KWARGS = {"color": "#07A9FF", "alpha": 0.4}
P_VALUE_KWARGS = {"color": "grey", "alpha": 1}

log = logging.getLogger(__name__)


class PlotLimit:
    def __init__(
        self,
        ts_dir: str,
        brazil_dir: str,
        TRUNC=False,
        TRUNC_SIZE=11,
        RANDOM=False,
        NME=False,
        RV_SIZE=4,
    ):
        """
        ts_dir
            Absolute path to the directory containing the test statistic files generated with and tested against signal
        brazil_dir
            Absolute path to the directory containing the test statistic files generated with no signal and tested against signal
        """
        self.RANDOM = RANDOM
        self.TRUNC = TRUNC
        self.TRUNC_SIZE = TRUNC_SIZE
        self.NME = NME
        f = glob.glob(ts_dir + "/*_wilks_fine.h5")[0]

        if self.NME:
            df = h5py.File(f, "r")
            s = df["m_bb"][:]
            s_scanned = s
            ts = df["t_m_bb"][:]
            if ts.shape[1] > 1:
                ts = ts[:, 0]
            df.close()

            self.signal_rate = m_prime_to_m(s)
            self.ts = ts

        else:
            df = h5py.File(f, "r")
            s = df["s"][:]
            s_scanned = s
            ts = df["t_s"][:]
            if ts.shape[1] > 1:
                ts = ts[:, 0]
            df.close()

            self.signal_rate = s_prime_to_s(s)
            self.ts = ts

        files_per_s = []

        for s_val in s_scanned[:]:
            files = glob.glob(ts_dir + f"/{s_val}_*.h5")
            files_per_s.append(files)

        t_crits = []
        t_crit_lows = []
        t_crit_his = []

        toys_per_scanned_s = []
        Es_per_scanned_s = []

        for file_list in files_per_s[:]:
            toy_ts_per_file = []
            toy_num_per_file = []
            toy_denom_per_file = []
            Es_per_file = []
            try:
                if self.RANDOM:
                    for FILE in file_list:
                        f = h5py.File(FILE, "r")
                        if "Es" in f.keys():
                            Es_per_file.extend(f["Es"][:])
                        else:
                            Es_per_file.extend([])
                        toy_ts_per_file.extend(
                            f["ts"][:] + np.random.normal(0, RV_SIZE, len(f["ts"][:]))
                        )
                        f.close()
                elif self.TRUNC:
                    for FILE in file_list:
                        f = h5py.File(FILE, "r")
                        if "Es" in f.keys():
                            Es_per_file.extend(f["Es"][:])
                        else:
                            Es_per_file.extend([])
                        toy_ts_per_file.extend(np.around(f["ts"][:], self.TRUNC_SIZE))
                        toy_num_per_file.extend(
                            np.around(f["ts_num"][:], self.TRUNC_SIZE)
                        )
                        toy_denom_per_file.extend(
                            np.around(f["ts_denom"][:], self.TRUNC_SIZE)
                        )
                        f.close()
                else:
                    for FILE in file_list:
                        f = h5py.File(FILE, "r")
                        toy_ts_per_file.extend(f["ts"][:])
                        if "Es" in f.keys():
                            Es = f["Es"][:]
                            num = f["ts_num"][:]
                            denom = f["ts_denom"][:]
                        else:
                            Es = []
                            num = []
                            denom = []
                        Es_per_file.extend(Es)
                        toy_num_per_file.extend(num)
                        toy_denom_per_file.extend(denom)
                        f.close()

                tcrit_tuple, _ = ts_critical(
                    toy_ts_per_file, threshold=0.9, confidence=0.68, plot=False
                )
                t_crit, t_crit_low, t_crit_high = tcrit_tuple

                t_crits.append(t_crit)
                t_crit_lows.append(t_crit_low)
                t_crit_his.append(t_crit_high)

                toys_per_scanned_s.append(toy_ts_per_file)

                Es_per_scanned_s.append(Es_per_file)
            except:  # noqa: B001, E722
                t_crits.append(np.nan)
                t_crit_lows.append(np.nan)
                t_crit_his.append(np.nan)

        self.toys_per_scanned_s = toys_per_scanned_s
        self.Es_per_scanned_s = Es_per_scanned_s
        self.t_crits = t_crits
        self.t_crit_lows = t_crit_lows
        self.t_crit_his = t_crit_his

        self.p_values = get_p_values(toys_per_scanned_s, ts)

        brazil_files = glob.glob(brazil_dir + "/*.h5")

        toys_per_s_zero_sig = []
        num_per_s_zero_sig = []
        denom_per_s_zero_sig = []
        Es_per_s_zero_sig = []
        seeds_per_s_zero_sig = []
        for i in range(len(s[:])):
            toys_per_file = []
            num_per_file = []
            denom_per_file = []
            Es_per_file = []
            seeds_per_file = []

            if self.RANDOM:
                for file in brazil_files:
                    f = h5py.File(file, "r")
                    ts = f["ts"][:]
                    Es = f["Es"][:]
                    seeds = f["seed"][:]
                    toys_per_file.extend(
                        ts[i] + np.random.normal(0, RV_SIZE, len(ts[i]))
                    )
                    Es_per_file.extend(Es)
                    seeds_per_file.extend(seeds)

            elif self.TRUNC:
                for file in brazil_files:
                    f = h5py.File(file, "r")
                    ts = f["ts"][:]
                    Es = f["Es"][:]
                    seeds = f["seed"][:]
                    toys_per_file.extend(np.around(ts[i], TRUNC_SIZE))
                    denom_per_file.extend(np.around(f["ts_denom"][:][i], TRUNC_SIZE))
                    num_per_file.extend(np.around(f["ts_num"][:][i], TRUNC_SIZE))
                    Es_per_file.extend(Es)
                    seeds_per_file.extend(seeds)
            else:
                for file in brazil_files:
                    f = h5py.File(file, "r")
                    ts = f["ts"][:]
                    toys_per_file.extend(ts[i])
                    if "Es" in f.keys():
                        Es = f["Es"][:]
                        seeds = f["seed"][:]
                        num = f["ts_num"][:][i]
                        denom = f["ts_denom"][:][i]
                    else:
                        Es = []
                        seeds = []
                        denom = []
                        num = []

                    Es_per_file.extend(Es)
                    seeds_per_file.extend(seeds)
                    denom_per_file.extend(denom)
                    num_per_file.extend(num)
            toys_per_s_zero_sig.append(toys_per_file)
            num_per_s_zero_sig.append(num_per_file)
            denom_per_s_zero_sig.append(denom_per_file)
            Es_per_s_zero_sig.append(Es_per_file)
            seeds_per_s_zero_sig.append(seeds_per_file)

        self.toys_per_s_zero_sig = toys_per_s_zero_sig
        self.num_per_s_zero_sig = num_per_s_zero_sig
        self.denom_per_s_zero_sig = denom_per_s_zero_sig
        self.Es_per_s_zero_sig = Es_per_s_zero_sig
        self.seeds_per_s_zero_sig = seeds_per_s_zero_sig

        self.p_values_median, self.p_values_hi, self.p_values_lo = sensitivity(
            toys_per_s_zero_sig[:],
            toys_per_scanned_s[:],
            s_scanned[:],
            CL=erf(2 / np.sqrt(2)),
            plot=False,
        )
        self.p_values_median, self.p_values_hi_1, self.p_values_lo_1 = sensitivity(
            toys_per_s_zero_sig[:],
            toys_per_scanned_s[:],
            s_scanned[:],
            CL=erf(1 / np.sqrt(2)),
            plot=False,
        )

        # Some plotting defaults that the user can override before function calls
        # Defaults for critical ts plot
        self.critical_y_lim = (-0.25, 6.25)
        self.critical_x_lim = (-1e-2, self.signal_rate[-1] / 1e-25)

        # Defaults for sensitivity plot
        self.sensitivity_y_lim = (1e-3, 1.2)
        self.sensitivity_x_lim = (-1e-3, self.signal_rate[-1] / 1e-25)

        # Defaults for test statistic plot
        self.idx = 19
        self.RANGE = (-1, 10)

        # For NME sensitivity
        self.plot_nme_range = False
        self.nme_sensitivity_x_lim = (-1e-3, self.signal_rate[-1] * 1000)

    def plot_critical_test_statistic(self):
        if self.NME:
            s_approx = find_crossing(self.signal_rate, self.t_crits, self.ts)
            T_est = s_approx[-1]
            gammas = self.signal_rate
        else:
            gammas = self.signal_rate / 1e-25
            s_approx = find_crossing(gammas, self.t_crits, self.ts)
            T_est = 1 / (s_approx[-1] * 1e-25)

        fig, ax = plt.subplots(figsize=legendstyles.figsizes.JupyterNotebook)
        plt.axhline(y=2.71, label="Wilks' 90% CL", ls=":", **P_VALUE_KWARGS)

        plt.axvline(x=s_approx[0], label="90% CL crossing", ls=":", c="k")
        plt.axvline(x=s_approx[-1], ls=":", c="k")

        plt.plot(
            gammas,
            self.ts,
            ls="None",
            marker=".",
            color="k",
            label=r"$\tilde{t}_{\Gamma, \mathrm{ obs}}$",
        )
        plt.plot(
            gammas,
            self.t_crits,
            c=NICE_RED,
            label=r"$\tilde{t}_{\Gamma, 90\%\, \mathrm{ CL}}$",
        )
        plt.fill_between(
            gammas, self.t_crit_lows, self.t_crit_his, alpha=0.6, color=NICE_RED
        )

        if self.NME:
            plt.xlabel(r"$m_{\beta\beta}$ [eV]")
            plt.ylabel(r"$\tilde{t}_{m_{\beta\beta}}$")
            plt.plot(
                [],
                [],
                ls="None",
                label=r"limit $m_{\beta\beta} <$" + f"{s_approx[-1]: .2f} eV",
            )
        else:
            plt.xlabel(r"$\Gamma_{1/2}^{0\nu}$ [$10^{-25} \,\mathrm{yr}^{-1}$]")
            plt.ylabel(r"$\tilde{t}_{\Gamma}$")
            plt.plot(
                [],
                [],
                ls="None",
                label=rf"$\rm{{T}}_{{1/2}}^{{0\nu}} >${T_est: 0.2e} yr, 90% CL",
            )

        legendstyles.legend_watermark(ax, logo_suffix="-200", approved=True)
        # legendstyles.add_preliminary(ax, color="red")

        plt.ylim(*self.critical_y_lim)
        plt.xlim(*self.critical_x_lim)

        plt.legend()
        plt.tight_layout()

        return fig

    def plot_sensitivity(self):
        s_approx = find_crossing(self.signal_rate, self.p_values, 0.1)
        T_est = 1 / s_approx[-1]
        s_approx = find_crossing(self.signal_rate, self.p_values_median, 0.1)
        T_12_median = 1 / s_approx[-1]
        gammas = self.signal_rate / 1e-25

        fig, ax = plt.subplots(figsize=legendstyles.figsizes.JupyterNotebook)
        plt.plot(
            gammas,
            self.p_values,
            marker="s",
            markersize=3,
            color="k",
            label="obs., " + r"$\rm{T}_{1/2}^{0\nu} >$ " + f"{T_est: 0.2e} yr, 90% CL",
        )
        plt.plot(
            gammas,
            self.p_values_median,
            color="k",
            ls="--",
            label="median exp. bkg. only, "
            + r"$\rm{T}_{1/2}^{0\nu} >$ "
            + f"{T_12_median: 0.2e} yr, 90% CL",
        )

        plt.fill_between(
            gammas,
            self.p_values_lo,
            self.p_values_hi,
            **BRAZIL_2_SIGMA_KWARGS,
        )
        plt.fill_between(
            gammas,
            self.p_values_lo_1,
            self.p_values_hi_1,
            **BRAZIL_1_SIGMA_KWARGS,
        )

        log.warning(find_crossing(self.signal_rate, self.p_values_lo_1, 0.1))
        log.warning(find_crossing(self.signal_rate, self.p_values_hi_1, 0.1))
        # ------ get the legend-entries that are already attached to the axis
        colors, label = ax.get_legend_handles_labels()

        # ------ append the multicolor legend patches
        colors.append(
            MulticolorPatch(
                [
                    (BRAZIL_1_SIGMA_KWARGS["color"], BRAZIL_1_SIGMA_KWARGS["alpha"]),
                    (BRAZIL_2_SIGMA_KWARGS["color"], BRAZIL_2_SIGMA_KWARGS["alpha"]),
                ]
            )
        )
        label.append(r"median exp. bkg. only $\pm$ 1 $\sigma,\,\pm 2 \sigma$")

        plt.axhline(y=0.1, ls=":", **P_VALUE_KWARGS)
        plt.vlines(1 / T_est / 1e-25, 0, 1e-1, **P_VALUE_KWARGS)
        plt.vlines(1 / T_12_median / 1e-25, 0, 1e-1, ls="--", **P_VALUE_KWARGS)

        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.minorticks_on()

        plt.yscale("log")
        plt.xlabel(r"$\Gamma_{1/2}^{0\nu} \, [10^{-25} \,\mathrm{yr}^{-1}]$ ")
        plt.ylabel(r"$p$-value")

        # plt.legend()
        ax.legend(
            colors,
            label,
            loc="best",
            handler_map={MulticolorPatch: MulticolorPatchHandler()},
            frameon=False,
        )
        legendstyles.legend_watermark(ax, logo_suffix="-200", approved=True)
        # legendstyles.add_preliminary(ax, color="red")

        plt.ylim(*self.sensitivity_y_lim)
        plt.xlim(*self.sensitivity_x_lim)
        plt.tight_layout()
        return fig

    def plot_test_statistic(self):
        RANGE = self.RANGE

        plt.figure(figsize=(12, 8))
        n, bins, _ = plt.hist(
            self.toys_per_scanned_s[self.idx], range=RANGE, bins=1000, color="orange"
        )
        n2, bins2, _ = plt.hist(
            self.toys_per_s_zero_sig[self.idx], range=RANGE, bins=1000, color="blue"
        )
        plt.clf()

        fig, ax = plt.subplots(figsize=PLTSIZE)
        gammas = self.signal_rate / 1e-25

        dummy_ts = np.linspace(RANGE[0], RANGE[-1], 100)
        plt.plot(
            dummy_ts,
            chi2.pdf(dummy_ts, df=1)
            * len(self.toys_per_scanned_s[self.idx])
            * (bins[1] - bins[0]),
            ls=":",
            color="grey",
            zorder=2,
            label=r"$\chi^2$ ndof=1",
        )

        plt.axvline(
            np.median(self.toys_per_s_zero_sig[self.idx]),
            label=r"median $\Gamma_0$",
            color="k",
            ls="--",
            zorder=2,
        )
        CL = erf(1 / np.sqrt(2))
        upper_ts = np.quantile(
            self.toys_per_s_zero_sig[self.idx], 0.5 + CL / 2, method="linear"
        )
        lower_ts = np.quantile(
            self.toys_per_s_zero_sig[self.idx], 0.5 - CL / 2, method="linear"
        )
        plt.axvline(upper_ts, label=r"1$\sigma$", ls=":", color="k", alpha=1, zorder=2)
        plt.axvline(lower_ts, ls=":", color="k", alpha=1, zorder=2)

        plt.stairs(
            n2,
            bins2,
            color=legendstyles.colors.AchatBlue,
            label=rf"$\Gamma$'={gammas[self.idx]*1e-25:.4e}|$\Gamma$=0",
            alpha=1,
            zorder=100,
        )
        plt.stairs(
            n,
            bins,
            color=GOOD_ORANGE,
            label=rf"$\Gamma$'={gammas[self.idx]*1e-25:.4e}|$\Gamma$={gammas[self.idx]*1e-25:.4e}",
            alpha=1,
            zorder=100,
        )
        plt.xlabel(r"$\tilde{t}_{\Gamma}$")
        plt.ylabel("counts")
        plt.yscale("log")

        legendstyles.legend_watermark(ax, logo_suffix="-200", approved=True)
        # legendstyles.add_preliminary(ax, color="red")

        plt.legend()
        plt.xlim(*RANGE)

        median_ts = np.median(self.toys_per_s_zero_sig[self.idx])
        log.debug(median_ts)
        ordered_ts = np.sort(self.toys_per_scanned_s[self.idx])
        log.debug((len(ordered_ts[ordered_ts >= median_ts])) / len(ordered_ts))
        plt.tight_layout()

        return fig

    def plot_NME_sensitivity(self):  # noqa: N802
        s_approx = find_crossing(self.signal_rate, self.p_values, 0.1)
        T_est = s_approx[-1]
        s_approx = find_crossing(self.signal_rate, self.p_values_median, 0.1)
        T_12_median = s_approx[-1]
        m_bb = self.signal_rate * 1000

        fig, ax = plt.subplots(figsize=legendstyles.figsizes.JupyterNotebook)
        if self.plot_nme_range:
            (obs,) = ax.plot(
                m_bb,
                self.p_values,
                color="k",
            )
        else:
            (obs,) = ax.plot(m_bb, self.p_values, color="k", marker="s", markersize=3)

        plt.plot(
            m_bb,
            self.p_values_median,
            color="k",
            alpha=1,
            ls="--",
            label="median exp. bkg. only, "
            + r"$m_{\beta\beta} <$ "
            + f"{round(T_12_median*1000,-1): 0.0f} meV, 90% CL",
        )

        plt.fill_between(
            m_bb,
            self.p_values_lo,
            self.p_values_hi,
            **BRAZIL_2_SIGMA_KWARGS,
        )
        plt.fill_between(
            m_bb,
            self.p_values_lo_1,
            self.p_values_hi_1,
            **BRAZIL_1_SIGMA_KWARGS,
        )
        log.warning(find_crossing(self.signal_rate, self.p_values_lo_1, 0.1))
        log.warning(find_crossing(self.signal_rate, self.p_values_hi_1, 0.1))

        if self.plot_nme_range:
            m_bb_belley_low = T_est * 1000 * NME_central / (NME_central + NME_unc_hi)
            m_bb_belley_high = T_est * 1000 * NME_central / (NME_central - NME_unc_lo)
            m_bb_pheno_low = (NME_central / (NME_PHENO_LOW)) * T_est * 1000
            m_bb_pheno_high = (NME_central / (NME_PHENO_HIGH)) * T_est * 1000

            plt.plot(
                m_bb * NME_central / (NME_central + NME_unc_hi),
                self.p_values,
                color=GOOD_MAGENTA,
                alpha=0.5,
            )
            plt.plot(
                m_bb * NME_central / (NME_central - NME_unc_lo),
                self.p_values,
                color=GOOD_MAGENTA,
                alpha=0.5,
            )
            # plot the Belley
            band = ax.fill_between(
                np.linspace(m_bb_belley_low, m_bb_belley_high, 100),
                0.08,
                0.12,
                color=GOOD_MAGENTA,
                hatch="\\\\",
                alpha=0.8,
                facecolor="None",
                linewidth=2,
            )

            plt.vlines(m_bb_belley_low, 0, 1e-1, color=GOOD_MAGENTA, alpha=0.8)
            plt.vlines(m_bb_belley_high, 0, 1e-1, color=GOOD_MAGENTA, alpha=0.8)

            # plot the pheno
            plt.fill_between(
                np.linspace(m_bb_pheno_low, m_bb_pheno_high, 100),
                0.08,
                0.12,
                color=GOOD_ORANGE,
                hatch="//",
                alpha=0.8,
                facecolor="None",
                linewidth=2,
                label="phenom., "
                + r"$m_{\beta\beta} <$ "
                + rf"${round(m_bb_pheno_high, -1): 0.0f}-{round(m_bb_pheno_low, -1): 0.0f}$ meV, 90% CL, "
                + rf"NME: {NME_PHENO_HIGH}- {NME_PHENO_LOW}",
            )

            plt.vlines(m_bb_pheno_low, 0, 1e-1, color=GOOD_ORANGE, alpha=0.8)
            plt.vlines(m_bb_pheno_high, 0, 1e-1, color=GOOD_ORANGE, alpha=0.8)

        # ------ get the legend-entries that are already attached to the axis
        colors, label = ax.get_legend_handles_labels()

        # ------ append the multicolor legend patches
        colors.append(
            MulticolorPatch(
                [
                    (BRAZIL_1_SIGMA_KWARGS["color"], BRAZIL_1_SIGMA_KWARGS["alpha"]),
                    (BRAZIL_2_SIGMA_KWARGS["color"], BRAZIL_2_SIGMA_KWARGS["alpha"]),
                ]
            )
        )
        label.append(r"median exp. bkg. only $\pm$ 1 $\sigma,\,\pm 2 \sigma$")

        if self.plot_nme_range:
            colors.append((band, obs))
            label.append(
                "obs., "
                + r"$m_{\beta\beta} <$ "
                + rf"${round(T_est*1000, -1): 0.0f}_{{-{round(T_est*1000-m_bb_belley_low, -1)}}}^{{+{round(m_bb_belley_high-T_est*1000, -1)}}}$ meV, 90% CL, "
                + rf"$\mathrm{{NME}}: {NME_central}_{{-{NME_unc_lo}}}^{{+{NME_unc_hi}}}$"
            )
            plt.axhline(y=0.1, ls=":", color="k")
            plt.vlines(T_est * 1000, 0, 1e-1, color="k")
            plt.vlines(T_12_median * 1000, 0, 1e-1, ls="--", color="k")
        else:
            colors.append(obs)
            label.append(
                "obs., "
                + r"$m_{\beta\beta} <$ "
                + rf"${round(T_est*1000, -1): 0.0f}$ meV, 90% CL"
            )

            plt.axhline(y=0.1, ls=":", **P_VALUE_KWARGS)
            plt.vlines(T_est * 1000, 0, 1e-1, **P_VALUE_KWARGS)
            plt.vlines(T_12_median * 1000, 0, 1e-1, ls="--", **P_VALUE_KWARGS)

        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.minorticks_on()

        plt.yscale("log")
        plt.xlabel(r"$m_{\beta\beta}$  [meV]")
        plt.ylabel(r"$p$-value")

        # plt.legend()
        ax.legend(
            colors,
            label,
            loc="best",
            handler_map={MulticolorPatch: MulticolorPatchHandler()},
            frameon=False,
        )
        legendstyles.legend_watermark(ax, logo_suffix="-200", approved=True)
        # legendstyles.add_preliminary(ax, color="red")

        plt.ylim(*self.sensitivity_y_lim)
        plt.xlim(*self.nme_sensitivity_x_lim)
        plt.tight_layout()
        return fig
