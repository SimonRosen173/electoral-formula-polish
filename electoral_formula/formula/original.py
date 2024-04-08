from typing import Dict, Optional, Callable

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import math

from pandas import Series, DataFrame

from electoral_formula.formula.common import droop_quota, dict_argmax
from electoral_formula.common_static import REGS


#########
# NOTES
# 1. The code for "NationalFormulaOriginal" was copy and pasted from "NationalFormulaAmended" and code was
# modified or removed where appropriate. This code represents the electoral formula represented in the
# Electoral Act and all its passed modifications (which the electoral amendment bill adds to).

# 2. When numbers or letters are referenced it is with regard to portions of the Electoral Amendment Bill
# specifically version B1D - 2022. While the referencing refers to the amendment bill, these references
# can be updated to reference the proper sections of the Electoral Acts and its passed amendments. While not
# technically accurate the referencing is valuable in showing the flow of the algorithm
#########

class NationalFormulaOriginal:
    def __init__(
            self,
            reg_bal_dfs,
            reg_party_sizes,
            comp_party_sizes,
            reg_tot_seats: Dict[str, int],
            nat_tot_seats=400
    ):
        # reg_bal_dfs[key] -> columns: party (index), votes, is_ind, party_size
        self.reg_bal_dfs: Dict[str, DataFrame] = reg_bal_dfs
        self.agr_reg_bal_df = None
        self.reg_party_sizes: Dict[str, Series] = reg_party_sizes
        self.comp_party_sizes: Series = comp_party_sizes

        self.regions = list(reg_bal_dfs.keys())

        self.reg_tot_seats = reg_tot_seats
        self.nat_tot_seats = nat_tot_seats
        self.comp_tot_seats = nat_tot_seats - sum(reg_tot_seats.values())

        self.reg_dfs = {}
        self.comp_df = None
        self.agr_reg_df = None
        self.tot_df = None

        self._init_reg_dfs()
        self._init_comp_df()

        # Stats
        self.n_reg_calcs = 0
        self.n_comp_calcs = 0
        self.reg_quotas = {reg: [] for reg in self.regions}
        self.comp_quotas = []

        self.reg_stats_dfs = None
        self.comp_stats_df = None

        self.reg_stats = None
        self.comp_stats = None

    ##########
    #  INIT  #
    ##########

    def _init_reg_dfs(self):
        self.reg_dfs = {}
        for reg, df in self.reg_bal_dfs.items():
            reg_df = df.copy()
            reg_df["party_size"] = self.reg_party_sizes[reg]
            reg_df["tot_rem_seats"] = 0
            reg_df["tot_surplus"] = 0
            self.reg_dfs[reg] = reg_df

        self._calc_agr_reg_bal_df()

    def _init_comp_df(self):
        self.comp_df = self.agr_reg_bal_df.copy()
        self.comp_df["tot_rem_seats"] = 0
        self.comp_df["tot_surplus"] = 0
        self.comp_df["party_size"] = self.comp_party_sizes

    def _init_last_reg_forfeit(self):
        self.last_reg_forfeit = {reg: False for reg in self.reg_dfs.keys()}

    ##########

    def _calc_agr_reg_bal_df(self):
        reg_bal_dfs = self.reg_bal_dfs
        first_reg = list(reg_bal_dfs.keys())[0]
        agr_reg_bal_df = reg_bal_dfs[first_reg].copy()
        for reg, df in reg_bal_dfs.items():
            agr_reg_bal_df["votes"] += df["votes"]

        self.agr_reg_bal_df = agr_reg_bal_df

    ###################
    # CALCULATE SEATS #
    ###################
    def calc_seats(self):
        self._calc_reg_seats()
        self._calc_comp_seats()

    # REGIONAL SEATS #
    def _calc_reg_seats(self):
        self.n_reg_calcs = 1
        self._first_calc_reg_seats()
        # If any regions forfeited a seat in last calc/recalc
        is_reg_forfeits = self._forf_reg_seats()
        while True in is_reg_forfeits.values():
            self.n_reg_calcs += 1
            self._recalc_reg_seats(is_reg_forfeits)
            is_reg_forfeits = self._forf_reg_seats()

        self._calc_agr_reg_df()

    def _first_calc_reg_seats(self):
        reg_dfs: Dict[str, DataFrame] = self.reg_dfs
        reg_tot_seats = self.reg_tot_seats
        is_reg_forfeits = {reg: False for reg in reg_dfs.keys()}
        # self._init_last_reg_forfeit()

        # Item 5
        for reg, reg_df in reg_dfs.items():
            tot_seats = reg_tot_seats[reg]
            # (a) & (b)
            reg_quota = droop_quota(tot_votes=reg_df["votes"].sum(), tot_seats=tot_seats)
            self.reg_quotas[reg].append(reg_quota)

            # (c)
            reg_df["seats"] = reg_df["votes"] // reg_quota

            # (d) & (e)
            # Surplus votes
            reg_df["surplus"] = reg_df["votes"] - (reg_df["seats"] * reg_quota)
            reg_df["tot_surplus"] += reg_df["surplus"]
            reg_df.sort_values(by="surplus", ascending=False, inplace=True)
            n_rem_seats = tot_seats - reg_df["seats"].sum()
            reg_df.iloc[:n_rem_seats, reg_df.columns.get_loc("seats")] += 1
            reg_df.iloc[:n_rem_seats, reg_df.columns.get_loc("tot_rem_seats")] += 1
            # assert reg_df["seats"].sum() == tot_seats, f"reg_df['seats'].sum() = {reg_df['seats'].sum()} != tot_seats = {tot_seats}"

            reg_df["forf_seats"] = 0
            reg_df["is_forfeit"] = False

    def _recalc_reg_seats(self, last_reg_forfeit):
        reg_dfs = self.reg_dfs
        # 7.3.
        for reg, reg_df in reg_dfs.items():
            if last_reg_forfeit[reg]:
                # forf_cond = reg_df["is_forf"] == True
                # (a)
                recalc_df = reg_df[reg_df["is_forfeit"] == False].copy()
                assigned_seats = reg_df.loc[reg_df["is_forfeit"] == True, "seats"].sum()

                # (b) & (c)
                recalc_tot_votes = recalc_df["votes"].sum()
                recalc_tot_seats = self.reg_tot_seats[reg] - assigned_seats
                recalc_quota = droop_quota(recalc_tot_votes, recalc_tot_seats)
                self.reg_quotas[reg].append(recalc_quota)

                # (d)
                recalc_df["seats"] = recalc_df["votes"]//recalc_quota

                # (e) & (f)
                recalc_df["surplus"] = recalc_df["votes"] - recalc_df["seats"] * recalc_quota
                recalc_df["tot_surplus"] += recalc_df["surplus"]
                recalc_df.sort_values(by="surplus", ascending=False, inplace=True)
                n_rem_seats = recalc_tot_seats - recalc_df["seats"].sum()
                recalc_df.iloc[:n_rem_seats, recalc_df.columns.get_loc("seats")] += 1
                recalc_df.iloc[:n_rem_seats, recalc_df.columns.get_loc("tot_rem_seats")] += 1

                for column in recalc_df.columns:
                    reg_df.loc[recalc_df.index, column] = recalc_df[column]
            else:
                # Don't need to do recalc if no seats were forfeited last calc for this region
                assert reg_df["seats"].sum() == self.reg_tot_seats[
                    reg], f"{reg} => {reg_df['seats'].sum()} != {self.reg_tot_seats[reg]}"

    # Forfeit seats in regional dfs according to 5. (f) & (g) or 7. (1) & (2)
    def _forf_reg_seats(self):
        reg_dfs: Dict[str, DataFrame] = self.reg_dfs
        is_forf_seats = {reg: False for reg in reg_dfs.keys()}

        for reg, reg_df in reg_dfs.items():
            # 1. Forfeit if number of seats won exceeds party list
            forf_cond_1 = (reg_df["seats"] > reg_df["party_size"])
            if forf_cond_1.any():
                is_forf_seats[reg] = True
                # reg_df.loc[forf_cond_1, "is_forf"] = True
                reg_df.loc[forf_cond_1, "forf_seats"] += reg_df.loc[forf_cond_1, "seats"] \
                                                         - reg_df.loc[forf_cond_1, "party_size"]
                reg_df.loc[forf_cond_1, "seats"] += reg_df.loc[forf_cond_1, "party_size"]

        # Find parties who forfeited seats
        for reg, reg_df in reg_dfs.items():
            reg_df["is_forfeit"] = reg_df["forf_seats"] > 0

        return is_forf_seats

    def _calc_agr_reg_df(self):
        reg_dfs = self.reg_dfs
        agr_reg_df = reg_dfs["gp"][["votes", "seats", "forf_seats", "tot_rem_seats", "tot_surplus",
                                    "is_forfeit"]].copy()
        for reg, reg_df in reg_dfs.items():
            if reg != "gp":
                for col in ["votes", "seats", "forf_seats", "tot_rem_seats", "tot_surplus"]:
                    agr_reg_df[col] += reg_df[col]

                agr_reg_df["is_forfeit"] = agr_reg_df["is_forfeit"] & reg_df["is_forfeit"]

        self.agr_reg_df = agr_reg_df

    # COMPENSATORY SEATS #
    def _calc_comp_seats(self):
        self.n_comp_calcs = 1
        self._init_tot_df()
        self._first_calc_comp_seats()
        is_forf = self._forf_comp_seats()
        while is_forf:
            self.n_comp_calcs += 1
            self._recalc_comp_seats()
            is_forf = self._forf_comp_seats()

    def _first_calc_comp_seats(self):
        agr_reg_df = self.agr_reg_df

        # 6.
        tot_df = self.tot_df

        # (a)
        tot_votes = tot_df["votes"].sum()
        tot_seats = self.nat_tot_seats
        quota = droop_quota(tot_votes, tot_seats)
        self.comp_quotas.append(quota)

        # (b)
        tot_df["seats"] = tot_df["votes"] // quota

        # (c)
        tot_df["surplus"] = tot_df["votes"] - tot_df["seats"] * quota
        self.comp_df.loc[tot_df.index, "tot_surplus"] += tot_df["surplus"]
        tot_df.sort_values(by="surplus", ascending=False, inplace=True)
        n_rem_seats = tot_seats - tot_df["seats"].sum()
        if n_rem_seats > 5:
            tot_df.iloc[:5, tot_df.columns.get_loc("seats")] += 1
            self.comp_df.loc[tot_df.iloc[:5].index, "tot_rem_seats"] += 1
            other_seats = n_rem_seats - 5
            tot_df["votes_per_seat"] = tot_df["votes"]/tot_df["seats"]
            tot_df.sort_values(by="votes_per_seat", ascending=False, inplace=True)
            tot_df.iloc[:other_seats, tot_df.columns.get_loc("seats")] += 1
            self.comp_df.loc[tot_df.iloc[:other_seats].index, "tot_rem_seats"] += 1
        else:
            tot_df.iloc[:n_rem_seats, tot_df.columns.get_loc("seats")] += 1
            self.comp_df.loc[tot_df.iloc[:n_rem_seats].index, "tot_rem_seats"] += 1
            tot_df["votes_per_seat"] = np.NaN

        # (d)
        self.comp_df["seats"] = tot_df["seats"] - agr_reg_df["seats"]
        # assert self.comp_df["seats"].sum() == self.comp_tot_seats, \
        #     f"comp_df['seats'] sums to wrong number {self.comp_df['seats'].sum()} != {self.comp_tot_seats}"

        tot_df["is_forfeit"] = False
        tot_df["forf_seats"] = 0
        self.comp_df["is_forfeit"] = False
        self.comp_df["forf_seats"] = 0

    def _recalc_comp_seats(self):
        # 7. (4)
        agr_reg_df = self.agr_reg_df
        tot_df = self.tot_df

        # (a)
        recalc_df = tot_df[tot_df["is_forfeit"] == False].copy()

        # (b) & (c)
        tot_votes = recalc_df["votes"].sum()
        tot_seats = self.nat_tot_seats - tot_df.loc[tot_df["is_forfeit"] == True, "seats"].sum()
        quota = droop_quota(tot_votes, tot_seats)
        self.comp_quotas.append(quota)

        # (d)
        recalc_df["seats"] = recalc_df["votes"] // quota

        # (e)
        recalc_df["surplus"] = recalc_df["votes"] - recalc_df["seats"] * quota
        recalc_df.sort_values(by="surplus", ascending=False, inplace=True)
        n_rem_seats = tot_seats - recalc_df["seats"].sum()
        if n_rem_seats > 5:
            recalc_df.iloc[:5, recalc_df.columns.get_loc("seats")] += 1
            self.comp_df.loc[recalc_df.iloc[:5].index, "tot_rem_seats"] += 1
            other_seats = n_rem_seats - 5
            recalc_df["votes_per_seat"] = recalc_df["votes"] / recalc_df["seats"]
            recalc_df.sort_values(by="votes_per_seat", ascending=False, inplace=True)
            recalc_df.iloc[:other_seats, recalc_df.columns.get_loc("seats")] += 1
            self.comp_df.loc[recalc_df.iloc[:other_seats].index, "tot_rem_seats"] += 1
        else:
            recalc_df.iloc[:n_rem_seats, recalc_df.columns.get_loc("seats")] += 1
            self.comp_df.loc[recalc_df.iloc[:n_rem_seats].index, "tot_rem_seats"] += 1
            recalc_df["votes_per_seat"] = np.NaN

        # (f)
        for col in recalc_df.columns:
            tot_df.loc[recalc_df.index, col] = recalc_df.loc[:, col]

        # ind_cond = agr_reg_df["is_ind"] == False
        self.comp_df["seats"] = tot_df["seats"] - agr_reg_df["seats"]
        assert self.comp_df["seats"].sum() == self.comp_tot_seats, \
            f"comp_df['seats'] sums to wrong number {self.comp_df['seats'].sum()} != {self.comp_tot_seats}"

    def _forf_comp_seats(self):
        is_forf = False
        comp_df = self.comp_df
        tot_df = self.tot_df
        agr_reg_df = self.agr_reg_df
        forf_cond = comp_df["seats"] > comp_df["party_size"]
        if forf_cond.any():
            is_forf = True
            forf_index = comp_df[forf_cond].index
            comp_df.loc[forf_cond, "is_forfeit"] = True
            forf_seats = comp_df.loc[forf_cond, "seats"] - comp_df.loc[forf_cond, "party_size"]
            comp_df.loc[forf_cond, "forf_seats"] += forf_seats
            comp_df.loc[forf_cond, "seats"] = comp_df.loc[forf_cond, "party_size"]

            tot_df.loc[forf_cond, "is_forfeit"] = True
            tot_df.loc[forf_cond, "forf_seats"] = tot_df.loc[forf_cond, "forf_seats"] + forf_seats
            tot_df.loc[forf_cond, "seats"] = agr_reg_df.loc[forf_index, "seats"] \
                                             + comp_df.loc[forf_cond, "seats"]

        return is_forf

    def _init_tot_df(self):
        self.tot_df = self.agr_reg_df[["votes"]].copy()

    ###################

    def calc_reg_stats(self):
        reg_dfs = self.reg_dfs
        reg_stats_dfs = {}
        reg_stats = {}

        for reg, reg_df in reg_dfs.items():
            df = reg_df.copy()
            df = df[(df["is_forfeit"]==False)]
            tot_votes = df["votes"].sum()
            tot_seats = df["seats"].sum()
            df["perc_votes"] = df["votes"]/tot_votes
            df["perc_seats"] = df["seats"]/tot_seats
            df["opt_seats"] = df["perc_votes"] * tot_seats
            df["diff_seats"] = (df["opt_seats"] - df["seats"]).abs()
            df["diff_perc"] = (df["perc_votes"] - df["perc_seats"]).abs()
            curr_stats = {
                "max_diff_seats": df["diff_seats"].max(),
                "mean_diff_seats": df["diff_seats"].mean(),
                "max_diff_perc": df["diff_perc"].max(),
                "mean_diff_perc": df["diff_perc"].mean(),
            }

            reg_stats[reg] = curr_stats
            reg_stats_dfs[reg] = df

        self.reg_stats_dfs = reg_stats_dfs
        self.reg_stats = reg_stats
        return reg_stats_dfs, reg_stats

    # This doesn't make sense as a metric
    def calc_comp_stats(self):
        comp_df = self.comp_df
        assert comp_df is not None
        stats_df = comp_df[(comp_df["is_forfeit"]==False)].copy()

        # stats_stats_df = stats_df[(stats_df["is_forfeit"] == False) & (stats_df["is_ind"] == False)]
        tot_votes = stats_df["votes"].sum()
        tot_seats = stats_df["seats"].sum()
        stats_df["perc_votes"] = stats_df["votes"] / tot_votes
        stats_df["perc_seats"] = stats_df["seats"] / tot_seats
        stats_df["opt_seats"] = stats_df["perc_votes"] * tot_seats
        stats_df["diff_seats"] = (stats_df["opt_seats"] - stats_df["seats"]).abs()
        stats_df["diff_perc"] = (stats_df["perc_votes"] - stats_df["perc_seats"]).abs()
        stats = {
            "max_diff_seats": stats_df["diff_seats"].max(),
            "mean_diff_seats": stats_df["diff_seats"].mean(),
            "max_diff_perc": stats_df["diff_perc"].max(),
            "mean_diff_perc": stats_df["diff_perc"].mean(),
        }

        return stats_df, stats

    def calc_nat_stats(self, exclude_forf=True):
        nat_df = self.calc_nat_df()
        if exclude_forf is False:
            raise NotImplementedError

        stats_df = nat_df[nat_df["is_forfeit"]==False].copy()

        tot_votes = stats_df["votes"].sum()
        tot_seats = stats_df["seats"].sum()
        stats_df["perc_votes"] = stats_df["votes"] / tot_votes
        stats_df["perc_seats"] = stats_df["seats"] / tot_seats
        stats_df["opt_seats"] = stats_df["perc_votes"] * tot_seats
        stats_df["diff_seats"] = stats_df["opt_seats"] - stats_df["seats"]
        stats_df["diff_perc"] = stats_df["perc_votes"] - stats_df["perc_seats"]
        stats_df["abs_diff_seats"] = (stats_df["opt_seats"] - stats_df["seats"]).abs()
        stats_df["abs_diff_perc"] = (stats_df["perc_votes"] - stats_df["perc_seats"]).abs()
        stats = {
            "max_diff_seats": stats_df["diff_seats"].max(),
            "mean_diff_seats": stats_df["diff_seats"].mean(),
            "min_diff_seats": stats_df["diff_seats"].min(),
            "max_diff_perc": stats_df["diff_perc"].max(),
            "mean_diff_perc": stats_df["diff_perc"].mean(),
            "min_diff_perc": stats_df["diff_perc"].min(),
        }

        return stats_df, stats

    def calc_nat_df(self):
        comp_df = self.comp_df
        nat_df = self.agr_reg_df[["votes", "seats", "forf_seats", "tot_rem_seats", "tot_surplus", "is_forfeit"]].copy()
        for col in ["seats", "forf_seats", "tot_rem_seats", "tot_surplus"]:
            nat_df[col] += comp_df[col]

        nat_df["is_forfeit"] = nat_df["is_forfeit"] & comp_df["is_forfeit"]
        return nat_df

    def save_reg_dfs(self, folder_path=None, file_path=None):
        if folder_path is None and file_path is None:
            raise ValueError("Either folder_path or file_path must be set")

        if folder_path is not None:
            for reg, reg_df in self.reg_dfs.items():
                file_path = f"{folder_path}/{reg}_seats.csv"
                reg_df.to_csv(file_path)

        if file_path is not None:
            reg_dfs = self.reg_dfs
            first_region = list(reg_dfs.keys())[0]
            amalg_df = reg_dfs[first_region].copy().reset_index()
            amalg_df["region"] = first_region

            for reg, reg_df in reg_dfs.items():
                if reg != first_region:
                    tmp_df = reg_df.copy().reset_index()
                    tmp_df["region"] = reg
                    amalg_df = pd.concat([amalg_df, tmp_df], ignore_index=True)

            amalg_df.to_csv(file_path, index=False)

    def save_to_excel(self, file_path):
        reg_dfs = self.reg_dfs
        comp_df = self.comp_df
        df = reg_dfs["gp"][["seats"]].copy()
        df.rename({"seats": "gp_seats"}, axis=1, inplace=True)
        for reg, reg_df in reg_dfs.items():
            df[f"{reg}_seats"] = reg_df["seats"]
        df.loc[comp_df.index, "comp_seats"] = comp_df.loc[comp_df.index, "seats"].copy()
        df["tot_seats"] = self.tot_df["seats"]
        assert df["tot_seats"].sum() == 400, "Sum of seats must be 400"
        df.to_excel(file_path)

    def save_comp_df(self, file_path):
        self.comp_df.to_csv(file_path)