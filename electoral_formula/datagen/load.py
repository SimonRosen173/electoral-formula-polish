from typing import Optional, List, Dict, Union

import pandas as pd
import numpy as np
import math

from electoral_formula.common_static import REGS, REG_TOT_SEATS


# Loads data from Excel docs and converts into correct format
class LoadGenerator:
    def __init__(
            self,
            reg_bal_excel: str,
            comp_bal_excel: Optional[str] = None,
            reg_tot_seats: Optional[Dict[str, int]] = None
    ):
        self.reg_bal_excel = reg_bal_excel
        self.comp_bal_excel = comp_bal_excel

        if reg_tot_seats is None:
            reg_tot_seats = REG_TOT_SEATS
        self.reg_tot_seats = reg_tot_seats

        # BALLOTS
        self.reg_bal_dfs: Optional[Dict[str, pd.DataFrame]] = None
        self.comp_bal_df: Optional[pd.DataFrame] = None
        self.agr_reg_bal_df: Optional[pd.DataFrame] = None

        # SIZES
        self.reg_party_sizes: Optional[Dict[str, pd.Series]] = None
        self.comp_party_sizes: Optional[pd.Series] = None

        # Load Regional Ballot\
        self._load_reg_ballots()

        # If applicable load comp ballot
        if comp_bal_excel is not None:
            self._load_comp_ballot()

    def _load_reg_ballots(self):
        all_df = pd.read_excel(self.reg_bal_excel, index_col="party")
        reg_bal_dfs = {}
        self.reg_party_sizes = {}
        for reg in REGS:
            reg_df = all_df[["is_ind"]].copy()
            reg_df["votes"] = all_df[f"{reg}_votes"].copy()
            # reg_df["party_size"] = all_df[f"{reg}_party_list"].copy()
            self.reg_party_sizes[reg] = all_df[f"{reg}_party_list"].copy()

            reg_bal_dfs[reg] = reg_df
        self.reg_bal_dfs = reg_bal_dfs

    def _load_comp_ballot(self):
        df = pd.read_excel(self.comp_bal_excel, index_col="party")
        self.comp_bal_df = df[["votes"]].copy()
        self.comp_party_sizes = df["party_list"].copy()


def main():
    reg_bal_path = r"C:\Users\simon\Documents\Work\IEC\Code\electoral-formula\electoral_formula\data\ballot\nat_2019_reg.xlsx"
    comp_bal_path = r"C:\Users\simon\Documents\Work\IEC\Code\electoral-formula\electoral_formula\data\ballot\nat_2019_comp.xlsx"
    gen = LoadGenerator(reg_bal_excel=reg_bal_path, comp_bal_excel=comp_bal_path)
    reg_bal_dfs = gen.reg_bal_dfs
    reg_party_sizes = gen.reg_party_sizes
    comp_bal_df = gen.comp_bal_df
    comp_party_sizes = gen.comp_party_sizes
    print(":P")


if __name__ == "__main__":
    main()

