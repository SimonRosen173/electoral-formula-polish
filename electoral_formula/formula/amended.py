from typing import Dict, Optional, Callable

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from pandas import Series, DataFrame

from electoral_formula.formula.common import droop_quota, dict_argmax


#########
# NOTES
# 1. When numbers or letters are referenced it is with regard to portions of the Electoral Amendment Bill
# specifically version B1D - 2022
#########
class NationalFormulaAmended:
    def __init__(
            self,
            reg_bal_dfs,
            comp_bal_df,
            reg_party_sizes,
            comp_party_sizes,
            reg_tot_seats: Dict[str, int],
            nat_tot_seats=400
    ):
        # reg_bal_dfs[key] -> columns: party (index), votes, is_ind, party_size
        self.reg_bal_dfs: Dict[str, DataFrame] = reg_bal_dfs
        self.agr_reg_bal_df = None
        # comp_bal_df -> columns: party (index), votes, party_size
        self.comp_bal_df: DataFrame = comp_bal_df
        self.reg_party_sizes: Dict[str, Series] = reg_party_sizes
        self.comp_party_sizes: Series = comp_party_sizes

        self.reg_tot_seats = reg_tot_seats
        self.nat_tot_seats = nat_tot_seats
        self.comp_tot_seats = nat_tot_seats - sum(reg_tot_seats.values())

        self.regions = list(self.reg_bal_dfs.keys())

        self.ind_names = None
        self.ind_tot_seats = None

        self.reg_dfs = {}
        self.comp_df = None
        self.agr_reg_df = None
        self.tot_df = None
        self.nat_df = None

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

        df = list(self.reg_bal_dfs.values())[0]
        self.ind_names = list(df[df["is_ind"] == True].index)
        self._init_agr_reg_bal_df()

    def _init_comp_df(self):
        self.comp_df = self.comp_bal_df.copy()
        # Only copying comp_bal_df to get correct indices - This isn't elegant, but it works

        self.comp_df["party_size"] = self.comp_party_sizes
        self.comp_df["tot_rem_seats"] = 0
        self.comp_df["tot_surplus"] = 0
        self.comp_df["votes"] = self.agr_reg_bal_df.loc[self.agr_reg_bal_df["is_ind"]==False, "votes"]

        # self.comp_df.drop("votes", axis=1, inplace=True)  # Remove votes as these are only for comp bal

    def _init_last_reg_forfeit(self):
        self.last_reg_forfeit = {reg: False for reg in self.reg_dfs.keys()}

    ##########

    def _init_agr_reg_bal_df(self):
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
        self.nat_df = self.calc_nat_df()

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
            reg_df["seats"] = reg_df["votes"]//reg_quota

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

    # TODO
    def _recalc_reg_seats(self, last_reg_forfeit):
        reg_dfs = self.reg_dfs
        # 7.
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

                # Update reg_df to recalc values
                # reg_df.loc[recalc_df.index]
                # tmp_1 = reg_df.loc[recalc_df.index]
                # tmp_2 = recalc_df

                for column in recalc_df.columns:
                    reg_df.loc[recalc_df.index, column] = recalc_df[column]
                # reg_df.loc[recalc_df.index] = recalc_df.copy()
            else:
                # Don't need to do recalc if no seats were forfeited last calc for this region
                pass
                # assert reg_df["seats"].sum() == self.reg_tot_seats[reg], f"{reg} => {reg_df['seats'].sum()} != {self.reg_tot_seats[reg]}"

    # Forfeit seats in regional dfs according to 5. (f) & (g) or 7. (1) & (2)
    def _forf_reg_seats(self):
        reg_dfs: Dict[str, DataFrame] = self.reg_dfs
        is_forf_seats = {reg: False for reg in reg_dfs.keys()}
        ind_forfs = {ind: False for ind in self.ind_names}

        for reg, reg_df in reg_dfs.items():
            # 1. Forfeit if number of seats won exceeds party list
            forf_cond_1 = (reg_df["is_ind"] == False) & (reg_df["seats"] > reg_df["party_size"])
            if forf_cond_1.any():
                is_forf_seats[reg] = True
                # reg_df.loc[forf_cond_1, "is_forf"] = True
                reg_df.loc[forf_cond_1, "forf_seats"] += reg_df.loc[forf_cond_1, "seats"] \
                                                         - reg_df.loc[forf_cond_1, "party_size"]
                reg_df.loc[forf_cond_1, "seats"] += reg_df.loc[forf_cond_1, "party_size"]

            # 2. Forfeit if independent has won more than 1 seat
            forf_cond_2 = (reg_df["is_ind"] == True) & (reg_df["seats"] > 1)
            if forf_cond_2.any():
                is_forf_seats[reg] = True
                reg_df.loc[forf_cond_2, "forf_seats"] += reg_df.loc[forf_cond_2, "seats"] - 1
                reg_df.loc[forf_cond_2, "seats"] = 1
                # reg_df.loc[forf_cond_2, "is_forfeit"] = True

            # Calc 'proportion of votes'/vote_perc for 3.
            reg_df["vote_perc"] = reg_df["votes"]/reg_df["votes"].sum()

        # 3. Forfeit if independent has won seats in multiple regions
        for ind in self.ind_names:
            ind_regs = {reg: reg_dfs[reg].loc[ind, ["seats", "vote_perc"]] for reg in reg_dfs.keys()}
            ind_regs = {key: val for key, val in ind_regs.items() if val["seats"] > 0}
            # ind_dict[ind] = {reg: reg_dfs[reg].loc[ind, ["seats", "votes"]] for reg in reg_dfs.keys()}
            # ind_dict[ind] = {key: val for key, val in ind_dict[ind].items() if val["seats"] > 0}

            max_reg = dict_argmax(ind_regs, key=lambda x: x["vote_perc"])
            other_regs = set(ind_regs.keys()) - {max_reg}

            for reg in other_regs:
                curr_df = reg_dfs[reg]
                curr_seats = curr_df.loc[ind, "seats"]
                if curr_seats > 0:
                    curr_df.loc[ind, "forf_seats"] += curr_seats
                    curr_df.loc[ind, "seats"] = 0
                    # curr_df.loc[ind, "is_forfeit"] = True
                    is_forf_seats[reg] = True

        # Find parties and independents who forfeited seats
        for reg, reg_df in reg_dfs.items():
            reg_df["is_forfeit"] = reg_df["forf_seats"] > 0
            for ind in self.ind_names:
                if reg_df.loc[ind, "is_forfeit"]:
                    ind_forfs[ind] = True

        for reg, reg_df in reg_dfs.items():
            for ind in self.ind_names:
                if ind_forfs[ind]:
                    reg_df.loc[ind, "is_forfeit"] = True

        return is_forf_seats

    def _calc_agr_reg_df(self):
        reg_dfs = self.reg_dfs
        agr_reg_df = reg_dfs["gp"][["votes", "seats", "forf_seats", "tot_rem_seats", "tot_surplus", "is_ind",
                                    "is_forfeit"]].copy()
        for reg, reg_df in reg_dfs.items():
            if reg != "gp":
                for col in ["votes", "seats", "forf_seats", "tot_rem_seats", "tot_surplus"]:
                    agr_reg_df[col] += reg_df[col]
                # agr_reg_df["votes"] += reg_df["votes"]
                # agr_reg_df["seats"] += reg_df["seats"]

                agr_reg_df["is_forfeit"] = agr_reg_df["is_forfeit"] & reg_df["is_forfeit"]

        self.agr_reg_df = agr_reg_df
        self.ind_tot_seats = agr_reg_df.loc[agr_reg_df["is_ind"] == True, "seats"].sum()

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
        tot_seats = self.nat_tot_seats - self.ind_tot_seats
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
        ind_cond = agr_reg_df["is_ind"] == False
        self.comp_df["seats"] = tot_df["seats"] - agr_reg_df.loc[ind_cond, "seats"]
        # assert self.comp_df["seats"].sum() == self.comp_tot_seats, \
        #     f"comp_df['seats'] sums to wrong number {self.comp_df['seats'].sum()} != {self.comp_tot_seats}"

        tot_df["is_forfeit"] = False
        tot_df["forf_seats"] = 0
        self.comp_df["is_forfeit"] = False
        self.comp_df["forf_seats"] = 0

    # TODO: Test
    def _recalc_comp_seats(self):
        # 7. (4)
        agr_reg_df = self.agr_reg_df
        tot_df = self.tot_df

        # (a)
        recalc_df = tot_df[tot_df["is_forfeit"] == False].copy()

        # (b) & (c)
        tot_votes = recalc_df["votes"].sum()
        tot_seats = self.nat_tot_seats - self.ind_tot_seats \
                    - tot_df.loc[tot_df["is_forfeit"] == True, "seats"].sum()
        quota = droop_quota(tot_votes, tot_seats)
        self.comp_quotas.append(quota)

        # (d)
        recalc_df["seats"] = recalc_df["votes"] // quota

        # (e)
        recalc_df["surplus"] = recalc_df["votes"] - recalc_df["seats"] * quota
        self.comp_df.loc[recalc_df.index, "tot_surplus"] += recalc_df["surplus"]
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

        ind_cond = agr_reg_df["is_ind"] == False
        self.comp_df["seats"] = tot_df["seats"] - agr_reg_df.loc[ind_cond, "seats"]
        assert self.comp_df["seats"].sum() == self.comp_tot_seats, \
            f"comp_df['seats'] sums to wrong number {self.comp_df['seats'].sum()} != {self.comp_tot_seats}"
        pass

    # TODO: Test
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
        agr_reg_df = self.agr_reg_df
        tot_df = self.comp_bal_df[["votes"]].copy()
        ind_cond = agr_reg_df["is_ind"] == False
        tot_df["votes"] += agr_reg_df.loc[ind_cond, "votes"]
        self.tot_df = tot_df

    ###################

    def calc_reg_stats(self):
        reg_dfs = self.reg_dfs
        reg_stats_dfs = {}
        reg_stats = {}

        for reg, reg_df in reg_dfs.items():
            df = reg_df.copy()
            df = df[(df["is_forfeit"]==False) & (df["is_ind"]==False)]
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

    def calc_nat_stats(self):
        # nat_df = self.calc_nat_df()
        nat_df = self.nat_df
        assert nat_df is not None
        cond = (nat_df["is_forfeit"]==False) & (nat_df["is_ind"]==False)
        stats_df = nat_df[cond].copy()

        tot_votes = stats_df["votes"].sum()
        tot_seats = stats_df["seats"].sum()
        stats_df["perc_votes"] = stats_df["votes"] / tot_votes
        stats_df["perc_seats"] = stats_df["seats"] / tot_seats
        stats_df["opt_seats"] = stats_df["perc_votes"] * tot_seats
        stats_df["diff_seats"] = stats_df["opt_seats"] - stats_df["seats"]
        stats_df["diff_perc"] = stats_df["perc_votes"] - stats_df["perc_seats"]
        stats_df["abs_diff_seats"] = (stats_df["opt_seats"] - stats_df["seats"]).abs()
        stats_df["abs_diff_perc"] = (stats_df["perc_votes"] - stats_df["perc_seats"]).abs()

        #################
        # FORFEIT SEATS #
        #################
        # Not sure if this is correct :)
        cond = (nat_df["is_forfeit"] == True) & (nat_df["is_ind"] == False)
        forf_stats_df = nat_df[cond].copy()

        forf_tot_votes = forf_stats_df["votes"].sum()
        forf_tot_seats = forf_stats_df["seats"].sum()
        curr_tot_votes = forf_tot_votes + tot_votes
        curr_tot_seats = forf_tot_seats + tot_seats

        forf_stats_df["perc_votes"] = forf_stats_df["votes"] / curr_tot_votes
        forf_stats_df["perc_seats"] = forf_stats_df["seats"] / curr_tot_seats
        forf_stats_df["opt_seats"] = forf_stats_df["perc_votes"] * curr_tot_seats
        forf_stats_df["diff_seats"] = forf_stats_df["opt_seats"] - forf_stats_df["seats"]
        forf_stats_df["diff_perc"] = forf_stats_df["perc_votes"] - forf_stats_df["perc_seats"]
        forf_stats_df["abs_diff_seats"] = (forf_stats_df["opt_seats"] - forf_stats_df["seats"]).abs()
        forf_stats_df["abs_diff_perc"] = (forf_stats_df["perc_votes"] - forf_stats_df["perc_seats"]).abs()
        #################

        #################
        #   IND SEATS   #
        #################
        # Not sure if this is correct :)
        cond = nat_df["is_ind"] == True
        ind_stats_df = nat_df[cond].copy()

        # ind_tot_votes = ind_stats_df["votes"].sum()
        # ind_tot_seats = ind_stats_df["seats"].sum()
        curr_tot_votes = nat_df["votes"].sum()
        curr_tot_seats = nat_df["seats"].sum()

        ind_stats_df["perc_votes"] = ind_stats_df["votes"] / curr_tot_votes
        ind_stats_df["perc_seats"] = ind_stats_df["seats"] / curr_tot_seats
        ind_stats_df["opt_seats"] = ind_stats_df["perc_votes"] * curr_tot_seats
        ind_stats_df["diff_seats"] = ind_stats_df["opt_seats"] - ind_stats_df["seats"]
        ind_stats_df["diff_perc"] = ind_stats_df["perc_votes"] - ind_stats_df["perc_seats"]
        ind_stats_df["abs_diff_seats"] = (ind_stats_df["opt_seats"] - ind_stats_df["seats"]).abs()
        ind_stats_df["abs_diff_perc"] = (ind_stats_df["perc_votes"] - ind_stats_df["perc_seats"]).abs()
        #################
        stats = {
            "max_diff_seats": stats_df["diff_seats"].max(),
            "mean_diff_seats": stats_df["diff_seats"].mean(),
            "min_diff_seats": stats_df["diff_seats"].min(),
            "max_diff_perc": stats_df["diff_perc"].max(),
            "mean_diff_perc": stats_df["diff_perc"].mean(),
            "min_diff_perc": stats_df["diff_perc"].min(),
            "tot_party_seats": stats_df["seats"].sum(),
            "tot_party_votes": stats_df["votes"].sum(),

            "tot_forf_seats": forf_stats_df["seats"].sum(),
            "tot_forf_votes": forf_stats_df["votes"].sum(),

            "tot_ind_seats": ind_stats_df["seats"].sum(),
            "tot_ind_votes": ind_stats_df["votes"].sum(),

            "tot_seats": nat_df["seats"].sum(),
            "tot_votes": nat_df["votes"].sum()
        }

        if stats["tot_seats"] != stats["tot_party_seats"] + stats["tot_forf_seats"] + stats["tot_ind_seats"]:
            raise ValueError(f"tot_seats != tot_party_seats + tot_forf_seats + tot_ind_seats. "
                             f"tot_seats={stats['tot_seats']}, tot_party_seats={stats['tot_party_seats']}, "
                             f"tot_forf_seats={stats['tot_forf_seats']}, tot_ind_seats={stats['tot_ind_seats']}")
        if stats["tot_votes"] != stats["tot_party_votes"] + stats["tot_forf_votes"] + stats["tot_ind_votes"]:
            raise ValueError(f"tot_votes != tot_party_votes + tot_forf_votes + tot_ind_votes. "
                             f"tot_votes={stats['tot_votes']}, tot_party_votes={stats['tot_party_votes']}, "
                             f"tot_forf_votes={stats['tot_forf_votes']}, tot_ind_votes={stats['tot_ind_votes']}")

        stats_df = pd.concat([stats_df, forf_stats_df, ind_stats_df], axis=0)  #, ignore_index=True)

        return stats_df, stats

    def calc_nat_df(self):
        comp_df = self.comp_df
        nat_df = self.agr_reg_df[["seats", "forf_seats", "tot_rem_seats", "tot_surplus", "is_ind", "is_forfeit"]].copy()
        nat_df["reg_votes"] = self.agr_reg_df["votes"]
        ind_cond = nat_df["is_ind"] == False
        for col in ["seats", "forf_seats", "tot_rem_seats", "tot_surplus"]:
            nat_df.loc[ind_cond, col] += comp_df[col]

        # nat_df.loc[ind_cond, "seats"] += comp_df["seats"]
        nat_df["comp_votes"] = 0
        nat_df.loc[ind_cond, "comp_votes"] = self.comp_bal_df["votes"]
        nat_df.loc[ind_cond, "is_forfeit"] = nat_df.loc[ind_cond, "is_forfeit"] & comp_df["is_forfeit"]
        nat_df["votes"] = nat_df["reg_votes"] + nat_df["comp_votes"]

        # Error check
        if (nat_df["seats"] < 0).any():
            raise ValueError("An issue occurred. Seats cannot be negative.")
        if (nat_df.loc[nat_df["is_ind"] == True, "seats"] > 1).any():
            raise ValueError("An issue occurred. An independent cannot get more than 1 seat.")

        if nat_df["seats"].sum() != 400:
            print(f"WARNING: Total seats awarded != 400. Total seats = {nat_df['seats'].sum()}")

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

    def save_comp_df(self, file_path):
        self.comp_df.to_csv(file_path)

    def save_to_excel(self, file_path):
        reg_dfs = self.reg_dfs
        comp_df = self.comp_df
        df = reg_dfs["gp"][["is_ind"]].copy()
        # df.rename({"seats": "gp_seats"}, axis=1, inplace=True)
        for reg, reg_df in reg_dfs.items():
            df[f"{reg}_seats"] = reg_df["seats"]
        df["comp_seats"] = 0
        df.loc[comp_df.index, "comp_seats"] = comp_df.loc[comp_df.index, "seats"]

        self._calc_agr_reg_df()
        df["tot_seats"] = self.agr_reg_df["seats"].copy()
        df.loc[comp_df.index, "tot_seats"] += comp_df.loc[comp_df.index, "seats"]

        assert df["tot_seats"].sum() == 400, "Sum of seats must be 400"
        df.to_excel(file_path)
