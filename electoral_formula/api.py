import json
import os
import sys
import pathlib
import argparse

sys.path.append('..')

from electoral_formula.utils import join_paths, create_or_clear_folder

PATH_SEP = os.path.sep
BASE_PATH_ARR = os.path.abspath(__file__).split(PATH_SEP)[:-1]

BASE_PATH = os.path.normpath(PATH_SEP.join(BASE_PATH_ARR))
CONFIGS_FOLDER = join_paths([BASE_PATH, "configs"])


def setup_run_folders(base_path):
    create_or_clear_folder(base_path)
    # Ballot
    create_or_clear_folder(join_paths([base_path, "ballot"]))
    # Gen Folder
    create_or_clear_folder(join_paths([base_path, "gen_data"]))
    # Reg Folder
    create_or_clear_folder(join_paths([base_path, "reg_data"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", type=str)
    parser.add_argument("--run", type=str)

    parser.add_argument("--join", type=str)

    parser.add_argument("-bn", "--batch-no", type=int)
    parser.add_argument("-il", "--is-local", action="store_true")
    parser.add_argument("-f", "--folder", type=str)
    parser.add_argument("-nb", "--n-batches", type=int)
    parser.add_argument("--formulae", type=str, choices=["amend", "orig", "amend_orig"])

    args = vars(parser.parse_args())
    if args["create"] is not None:
        from electoral_formula.cluster.slurmhandler import SlurmHandler
        sg = SlurmHandler(args["create"], is_local=args["is_local"])
        sg.create_slurm()
    elif args["run"] is not None:
        assert args["batch_no"] is not None
        assert args["folder"] is not None

        from electoral_formula.benchmarker import rand_exp, increasing_votes_exp

        setup_run_folders(args["folder"])

        config_path = join_paths([CONFIGS_FOLDER, args['run']])
        with open(config_path, "r") as f:
            config = json.load(f)

        exp_config = config["exp"]
        datagen_kwargs = config["datagen"]
        seed = exp_config["seed"] + args["batch_no"]

        if exp_config["type"] == "rand":
            if exp_config["formula"] == "amend_orig":
                use_original = True
            elif exp_config["formula"] == "amend":
                use_original = False
            else:
                raise ValueError(f'exp_config["formula"]={exp_config["formula"]} is an invalid value')

            rand_exp(
                n_runs=exp_config['n_runs'], batch_no=args["batch_no"], folder_path=args["folder"],
                data_gen_kwargs=datagen_kwargs, use_original=use_original, seed=seed
            )
        elif exp_config["type"] == "incr_votes":
            raise NotImplementedError
        else:
            raise ValueError(f"exp_config['type']={exp_config['type']} is an invalid value")
    elif args["join"] is not None:
        from electoral_formula.benchmarker import join_batch_data
        join_batch_data(int(args["n_batches"]), args["join"], args["formulae"])
    else:
        raise ValueError


if __name__ == "__main__":
    main()

