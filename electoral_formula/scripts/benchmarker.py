import os
import sys
import pathlib
import argparse

dir_path = str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-2]
dir_path = os.sep.join(dir_path)
sys.path.append(dir_path)

from electoral_formula.benchmarker import rand_exp, increasing_votes_exp


def main():
    parser = argparse.ArgumentParser()

    # exp
    parser.add_argument("-e", "--exp", choices=['rand', 'incr_votes'], required=True,
                        help="Specifies which experiment to run.")

    # folder
    parser.add_argument("-f", "--folder", required=True,
                        help="Folder in which results will be saved")
    # n_runs
    parser.add_argument("-nr", "--n_runs", type=int, required=True,
                        help="Number of simulations to be run")
    # formulas
    parser.add_argument("-fo", "--formula", required=True, choices=["amend", "orig", "amend_orig"],
                        help="Which formula/s to use")
    # comp-bal-from-reg
    parser.add_argument("-cbr", "--comp-bal-from-reg", action="store_true",
                        help="If true, compute compensatory ballot from regional otherwise comp ballot is random.")
    # n-parties
    parser.add_argument("-np", "--n-parties", type=int, required=True,
                        help="Number of parties")
    # n-inds
    parser.add_argument("-ni", "--n-inds", type=int, required=True,
                        help="Number of independents")
    # n-large-parties
    parser.add_argument("-nlp", "--n-large-parties", type=int, required=True,
                        help="Number of 'larger' parties")
    # perc-party-votes
    parser.add_argument("-ppv", "--perc-party-votes", type=float, required=True,
                        help="Percentage of votes reserved for parties")
    # perc-large-votes-min
    parser.add_argument("-plmin", "--perc-large-votes-min", type=float, required=True,
                        help="Minimum percentage of votes reservable for large parties.")
    # perc-large-votes-max
    parser.add_argument("-plmax", "--perc-large-votes-max", type=float, required=True,
                        help="Maximum percentage of votes reservable for large parties.")
    # repeats_per_vote
    parser.add_argument("-rpv", "--repeats_per_vote", type=int,
                        help="Number of repeated runs per specified votes. Only applicable for incr_votes exp")
    # start_perc
    parser.add_argument("-sp", "--start_perc", type=float, required=False,
                        help="Start percentage of votes assigned to party 1. Only applicable for incr_votes exp")
    # end_perc
    parser.add_argument("-ep", "--end_perc", type=float, required=False,
                        help="End percentage of votes assigned to party 1. Only applicable for incr_votes exp")

    args = vars(parser.parse_args())

    data_gen_kwargs = {
        "n_parties": args["n_parties"],
        "n_inds": args["n_inds"],
        "n_large_parties": args["n_large_parties"],
        "perc_reg_parties_votes": args["perc_party_votes"],
        "perc_large_votes_min": args["perc_large_votes_min"],
        "perc_large_votes_max": args["perc_large_votes_max"],
        "comp_bal_from_reg": args["comp_bal_from_reg"]
    }

    if args["exp"] == "rand":
        if args["formula"] == "amend_orig":
            use_orig = True
        elif args["formula"] == "amend":
            use_orig = False
        else:
            raise ValueError(f"formula={args['formula']} is not valid for rand exp. Only 'amend_orig' and 'amend' "
                             f"is supported")

        rand_exp(n_runs=args['n_runs'], folder_path=args['folder'], data_gen_kwargs=data_gen_kwargs,
                 use_original=use_orig, batch_no=0)
    elif args["exp"] == "incr_votes":
        if args['start_perc'] is None or args['end_perc'] is None or args['repeats_per_vote'] is None:
            raise ValueError("If exp is 'incr_votes' then 'start_perc', 'end_perc' and 'repeats_per_vote' must be "
                             "specified.")
        increasing_votes_exp(n_runs=args['n_runs'], repeats_per_run=args['repeats_per_vote'], batch_no=0,
                             folder_path=args['folder'], start_perc=args['start_perc'], end_perc=args['end_perc'],
                             data_gen_kwargs=data_gen_kwargs, formula=args['formula'])
    else:
        raise ValueError(f"exp = {args['exp']} is not supported")


if __name__ == "__main__":
    main()
