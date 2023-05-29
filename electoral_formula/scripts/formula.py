import os
import sys
import pathlib
import argparse

dir_path = str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-2]
dir_path = os.sep.join(dir_path)
sys.path.append(dir_path)

from electoral_formula.datagen import LoadGenerator, remove_inds
from electoral_formula.formula import NationalFormulaAmended, NationalFormulaOriginal


def main():
    desc = "CLI for calculating seats obtained by parties and independents using either the original " \
           "or amended electoral formulae given the required ballots."
    parser = argparse.ArgumentParser(description=desc)

    # reg_bal_path
    parser.add_argument("-rbp", "--reg-bal-path", required=True, type=str,
                        help="Path of excel doc containing regional ballot & party sizes")
    # comp_bal_path
    parser.add_argument("-cbp", "--comp-bal-path", required=True, type=str,
                        help="Path of excel doc containing compensatory ballot & party sizes. Note: when alg is"
                             "amend, votes can be set to anything in this doc as only party_list will be used.")
    # alg
    parser.add_argument("-a", "--alg", required=True, choices=["orig", "amend"],
                        help="Which algorithm to use to calculate seats")
    # seats_path
    parser.add_argument("-sp", "--seats-path", required=True, type=str,
                        help="Output path of excel file containing calculated seat allocations")

    # parser.add_argument("-if", "--ignore-forfeit", required=False, action="store_true",
    #                     help="Specify if parties/independents that forfeited seats must be excluded from stats"
    #                          "calculations.")
    # verbosity
    # parser.add_argument("-v", "--verbosity", required=False, default=0, choices=range(0, 2), type=int,
    #                     help="How much data and stats to include in outputted file. 0 - just seats. 1 - seats, "
    #                          "opt_seats & seat_diff. 2 - ?")

    # stats_path
    parser.add_argument("-stp", "--stats-path", required=True, type=str,
                        help="Output path of excel file containing stats for seat calculations")

    args = vars(parser.parse_args())

    # GET DATA
    assert args["reg_bal_path"][-5:] == ".xlsx", "reg_bal_path must be an excel spreadsheet and have extension .xlsx"
    if args["comp_bal_path"] is not None:
        assert args["comp_bal_path"][-5:] == ".xlsx", "comp_bal_path must be an excel spreadsheet and have extension .xlsx"

    assert args["seats_path"][-5:] == ".xlsx", "seats_path must be an excel spreadsheet and have extension .xlsx"

    print(f"###########")
    print(f"# BALLOTS #")
    print(f"###########")
    print(f"Regional Ballot Path = {args['reg_bal_path']}")
    print(f"Compensatory Ballot Path = {args['comp_bal_path']}")

    print(f"#########")
    print(f"# SEATS #")
    print(f"#########")
    print(f"Seats path = {args['seats_path']}")

    gen = LoadGenerator(reg_bal_excel=args["reg_bal_path"], comp_bal_excel=args["comp_bal_path"])

    alg = args["alg"]
    if alg == "orig":
        orig_reg_bal_dfs, orig_reg_party_sizes = remove_inds(
            reg_bal_dfs=gen.reg_bal_dfs,
            reg_party_sizes=gen.reg_party_sizes
        )
        nat_formula = NationalFormulaOriginal(reg_bal_dfs=orig_reg_bal_dfs,
                                              reg_party_sizes=orig_reg_party_sizes,
                                              comp_party_sizes=gen.comp_party_sizes,
                                              reg_tot_seats=gen.reg_tot_seats)
    elif alg == "amend":
        nat_formula = NationalFormulaAmended(reg_bal_dfs=gen.reg_bal_dfs,
                                             reg_party_sizes=gen.reg_party_sizes,
                                             comp_bal_df=gen.comp_bal_df,
                                             comp_party_sizes=gen.comp_party_sizes,
                                             reg_tot_seats=gen.reg_tot_seats
                                             )
    else:
        raise ValueError("Only 'amend' and 'orig' are supported values for alg")

    nat_formula.calc_seats()
    nat_formula.save_to_excel(args["seats_path"])

    stats_df, stats = nat_formula.calc_nat_stats()
    print(f"#########")
    print(f"# STATS #")
    print(f"#########")
    print(f"diff_seats:")
    print(f"\t min: {stats['min_diff_seats']}, mean: {stats['mean_diff_seats']}, max: {stats['max_diff_seats']}")
    print(f"diff_perc:")
    print(f"\t min: {stats['min_diff_perc']}, mean: {stats['mean_diff_perc']}, max: {stats['max_diff_perc']}")
    print(f"Stats path = {args['stats_path']}")

    stats_df.to_excel(args['stats_path'])


if __name__ == "__main__":
    main()
