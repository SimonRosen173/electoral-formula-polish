import os

import pandas as pd
import numpy as np
from tqdm import tqdm

from electoral_formula.common_static import REGS
from electoral_formula.formula import NationalFormulaOriginal, NationalFormulaAmended
from electoral_formula.datagen import RandomGenerator, remove_inds, df_to_str
from electoral_formula.utils import create_or_clear_folder, join_paths


def join_batch_data(n_batches, folder_path, formulas):
    regs = REGS

    out_folder = f"{folder_path}/agr"
    create_or_clear_folder(out_folder)
    # if not os.path.exists(out_folder):
    #     os.mkdir(out_folder)

    # file_names = {}
    if formulas == "amend":
        file_names = {
            "comp": f"comp_bal",
            "nat_amend": f"nat_amend"
        }
    elif formulas == "orig":
        file_names = {
            "comp": f"comp_bal",
            "nat_orig": f"nat_orig"
        }
    elif formulas == "amend_orig":
        file_names = {
            "comp": f"comp_bal",
            "nat_orig": f"nat_orig",
            "nat_amend": f"nat_amend",
            "nat_diff": f"nat_diff"
        }
    else:
        raise ValueError(f"formulas={formulas} is not supported")

    for reg in regs:
        file_names[reg] = f"reg_{reg}"

    # file_prefixes = {}
    # for key, val in file_names.items():
    #     file_prefixes[key] = f"{folder_path}/{val}"
    #
    # for reg in regs:
    #     file_prefixes[reg] = f"{folder_path}/reg_{reg}"
    #     file_names[reg] = f"reg_{reg}"

    dfs = {}
    for key, val in file_names.items():
        file_path = join_paths([folder_path, "run_0", f"{val}_0.csv"])
        try:
            dfs[key] = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error: {e}. \nkey: {key}\t path: {file_path}")

    print("Concating...")
    for i in tqdm(range(1, n_batches)):
        for key, val in file_names.items():
            file_path = join_paths([folder_path, f"run_{i}", f"{val}_{i}.csv"])
            curr_df = pd.read_csv(file_path)
            dfs[key] = pd.concat([dfs[key], curr_df], ignore_index=True)

    # for key, val in file_prefixes.items():
    #     try:
    #         dfs[key] = pd.read_csv(f"{val}_0.csv")
    #     except Exception as e:
    #         print(f"Error: {e}. \nkey:{key}\t val:{val}")
    #
    # print("Concating...")
    # for i in tqdm(range(1, n_batches)):
    #     for key, val in file_prefixes.items():
    #         curr_df = pd.read_csv(f"{val}_{i}.csv")
    #         dfs[key] = pd.concat([dfs[key], curr_df], ignore_index=True)

    print("Saving...")
    for key, file_name in file_names.items():
        dfs[key].to_csv(f"{out_folder}/{file_name}.csv", index=False)


def increasing_votes_exp(
    n_runs,
    repeats_per_run,
    batch_no,
    folder_path,
    start_perc,
    end_perc,
    data_gen_kwargs,
    formula
):
    regs = REGS

    tot_nat_amend_stats = {
        "max_diff_seats": 0,
        "mean_diff_seats": 0,
        "max_diff_perc": 0,
        "mean_diff_perc": 0
    }

    tot_nat_orig_stats = {
        "max_diff_seats": 0,
        "mean_diff_seats": 0,
        "max_diff_perc": 0,
        "mean_diff_perc": 0
    }

    tot_nat_diff_stats = {
        "max_diff_seats": 0,
        "mean_diff_seats": 0,
        "max_diff_perc": 0,
        "mean_diff_perc": 0
    }

    files_dict = {
        "comp": open(f"{folder_path}/comp_bal_{batch_no}.csv", "w"),
    }

    if formula == "amend" or formula == "amend_orig":
        is_amend = True
        files_dict["nat_amend"] = open(f"{folder_path}/nat_amend_{batch_no}.csv", "w")
    else:
        is_amend = False

    if formula == "orig" or formula == "amend_orig":
        is_orig = True
        files_dict["nat_orig"] = open(f"{folder_path}/nat_orig_{batch_no}.csv", "w")
    else:
        is_orig = False

    if formula == "amend_orig":
        files_dict["nat_diff"] = open(f"{folder_path}/nat_diff_{batch_no}.csv", "w")

    for reg in regs:
        files_dict[reg] = open(f"{folder_path}/reg_{reg}_{batch_no}.csv", "w")

    perc_step = (end_perc - start_perc) / n_runs
    curr_perc = start_perc

    n_runs += 1  # NB: So end_perc is included
    tot_iters = n_runs * repeats_per_run

    with tqdm(total=tot_iters) as prog_bar:
        curr_step = 0
        for run_no in range(n_runs):
            for repeat_no in range(repeats_per_run):
                rand_gen = RandomGenerator(
                    first_party_vote_perc=curr_perc,
                    **data_gen_kwargs
                )

                prog_bar_stats = {
                    "run": run_no,
                    "rep": repeat_no,
                    "perc": round(curr_perc * 100, 1)
                }

                if is_orig:
                    ########################
                    # ORIGINAL NAT FORMULA #
                    ########################
                    orig_reg_bal_dfs, orig_reg_party_sizes = remove_inds(
                        reg_bal_dfs=rand_gen.reg_bal_dfs,
                        reg_party_sizes=rand_gen.reg_party_sizes
                    )

                    nat_formula_orig = NationalFormulaOriginal(reg_bal_dfs=orig_reg_bal_dfs,
                                                               reg_party_sizes=orig_reg_party_sizes,
                                                               comp_party_sizes=rand_gen.comp_party_sizes,
                                                               reg_tot_seats=rand_gen.reg_tot_seats)
                    nat_formula_orig.calc_seats()

                    # Stats
                    nat_orig_stats_df, nat_orig_stats = nat_formula_orig.calc_nat_stats()
                    for key in nat_orig_stats.keys():
                        tot_nat_orig_stats[key] = max(tot_nat_orig_stats[key], nat_orig_stats[key])

                    # Write data
                    nat_orig_stats_df.reset_index(inplace=True)
                    if curr_step == 0:
                        nat_str = ",".join(nat_orig_stats_df.columns)
                        nat_str += "," + ",".join(nat_orig_stats.keys())
                        files_dict["nat_orig"].write(nat_str + "\n")
                    nat_str = df_to_str(nat_orig_stats_df)
                    nat_str += "," + ",".join([str(el) for el in nat_orig_stats.values()])
                    files_dict["nat_orig"].write(nat_str + "\n")

                    tot_nat_orig_stats["max_diff_seats"] = max(tot_nat_orig_stats["max_diff_seats"],
                                                               nat_orig_stats["max_diff_seats"])
                    prog_bar_stats["o_diff_seats"] = tot_nat_orig_stats["max_diff_seats"]

                if is_amend:
                    #######################
                    # AMENDED NAT FORMULA #
                    #######################
                    nat_formula_amend = NationalFormulaAmended(reg_bal_dfs=rand_gen.reg_bal_dfs,
                                                               comp_bal_df=rand_gen.comp_bal_df,
                                                               reg_party_sizes=rand_gen.reg_party_sizes,
                                                               comp_party_sizes=rand_gen.comp_party_sizes,
                                                               reg_tot_seats=rand_gen.reg_tot_seats)
                    nat_formula_amend.calc_seats()

                    # Stats
                    nat_amend_stats_df, nat_amend_stats = nat_formula_amend.calc_nat_stats()
                    for key in nat_amend_stats.keys():
                        tot_nat_amend_stats[key] = max(tot_nat_amend_stats[key], nat_amend_stats[key])

                    # Write data
                    nat_amend_stats_df.reset_index(inplace=True)
                    if curr_step == 0:
                        nat_str = ",".join(nat_amend_stats_df.columns)
                        nat_str += "," + ",".join(nat_amend_stats.keys())
                        files_dict["nat_amend"].write(nat_str + "\n")
                    nat_str = df_to_str(nat_amend_stats_df)
                    nat_str += "," + ",".join([str(el) for el in nat_amend_stats.values()])
                    files_dict["nat_amend"].write(nat_str + "\n")

                    # TODO: Progress bar
                    tot_nat_amend_stats["max_diff_seats"] = max(tot_nat_amend_stats["max_diff_seats"],
                                                                nat_amend_stats["max_diff_seats"])
                    prog_bar_stats["a_diff_seats"] = tot_nat_amend_stats["max_diff_seats"]

                if is_amend and is_orig:
                    ##################
                    # DIFF NAT STATS #
                    ##################
                    nat_diff_stats_df = nat_orig_stats_df.copy()
                    nat_diff_stats_df["votes"] -= nat_amend_stats_df["votes"]
                    nat_diff_stats_df["seats"] -= nat_amend_stats_df["seats"]

                    nat_diff_stats_df["perc_votes"] -= nat_amend_stats_df["perc_votes"]
                    nat_diff_stats_df["perc_seats"] -= nat_amend_stats_df["perc_seats"]

                    nat_diff_stats_df["diff_seats"] -= nat_amend_stats_df["diff_seats"]
                    nat_diff_stats_df["diff_perc"] -= nat_amend_stats_df["diff_perc"]

                    nat_diff_stats = {
                        "max_diff_seats": nat_diff_stats_df["diff_seats"].max(),
                        "mean_diff_seats": nat_diff_stats_df["diff_seats"].mean(),
                        "max_diff_perc": nat_diff_stats_df["diff_perc"].max(),
                        "mean_diff_perc": nat_diff_stats_df["diff_perc"].mean(),
                    }

                    for key in nat_diff_stats.keys():
                        tot_nat_diff_stats[key] = max(tot_nat_diff_stats[key], nat_diff_stats[key])

                    # Write data
                    nat_diff_stats_df.reset_index(inplace=True)
                    if curr_step == 0:
                        nat_str = ",".join(nat_diff_stats_df.columns)
                        nat_str += "," + ",".join(nat_diff_stats.keys())
                        files_dict["nat_diff"].write(nat_str + "\n")
                    nat_str = df_to_str(nat_diff_stats_df)
                    nat_str += "," + ",".join([str(el) for el in nat_diff_stats.values()])
                    files_dict["nat_diff"].write(nat_str + "\n")

                ####################
                # SAVE BALLOT DATA #
                ####################
                comp_bal_df = rand_gen.comp_bal_df
                comp_bal_df["party_size"] = rand_gen.comp_party_sizes
                comp_bal_df.reset_index(inplace=True)
                if curr_step == 0:
                    files_dict["comp"].write(",".join(comp_bal_df.columns) + "\n")
                files_dict["comp"].write(df_to_str(comp_bal_df) + "\n")

                for reg, reg_df in rand_gen.reg_bal_dfs.items():
                    reg_df["party_size"] = rand_gen.reg_party_sizes[reg]
                    reg_df.reset_index(inplace=True)
                    if curr_step == 0:
                        files_dict[reg].write(",".join(reg_df.columns) + "\n")
                    files_dict[reg].write(df_to_str(reg_df) + "\n")

                curr_step += 1
                prog_bar.update(1)
                prog_bar.set_postfix(prog_bar_stats)

            curr_perc += perc_step

    for file in files_dict.values():
        file.close()


def rand_exp(
        n_runs,
        batch_no,
        folder_path,
        data_gen_kwargs,
        use_original,
        seed
):
    np.random.seed(seed)

    regs = REGS
    tot_nat_amend_stats = {
        "max_diff_seats": 0,
        "mean_diff_seats": 0,
        "min_diff_seats": 0,
        "max_diff_perc": 0,
        "mean_diff_perc": 0,
        "min_diff_perc": 0,
    }

    tot_nat_orig_stats = {
        "max_diff_seats": 0,
        "mean_diff_seats": 0,
        "min_diff_seats": 0,
        "max_diff_perc": 0,
        "mean_diff_perc": 0,
        "min_diff_perc": 0
    }

    tot_nat_diff_stats = {
        "max_diff_seats": 0,
        "mean_diff_seats": 0,
        "min_diff_seats": 0,
        "max_diff_perc": 0,
        "mean_diff_perc": 0,
        "min_diff_perc": 0
    }

    files_dict = {
        "comp": open(f"{folder_path}/comp_bal_{batch_no}.csv", "w"),
        "nat_orig": open(f"{folder_path}/nat_orig_{batch_no}.csv", "w"),
        "nat_amend": open(f"{folder_path}/nat_amend_{batch_no}.csv", "w"),
        "nat_diff": open(f"{folder_path}/nat_diff_{batch_no}.csv", "w")
    }
    for reg in regs:
        files_dict[reg] = open(f"{folder_path}/reg_{reg}_{batch_no}.csv", "w")

    prog_bar = tqdm(range(n_runs))
    for i in prog_bar:
        rand_gen = RandomGenerator(
            **data_gen_kwargs
        )

        rand_gen.save_reg_bal_dfs(f"{folder_path}/ballot/ballot_{i}.csv")

        #######################
        # AMENDED NAT FORMULA #
        #######################
        nat_formula_amend = NationalFormulaAmended(reg_bal_dfs=rand_gen.reg_bal_dfs,
                                                   comp_bal_df=rand_gen.comp_bal_df,
                                                   reg_party_sizes=rand_gen.reg_party_sizes,
                                                   comp_party_sizes=rand_gen.comp_party_sizes,
                                                   reg_tot_seats=rand_gen.reg_tot_seats)
        nat_formula_amend.calc_seats()

        # Stats
        nat_amend_stats_df, nat_amend_stats = nat_formula_amend.calc_nat_stats()
        # for key in nat_amend_stats.keys():
        #     tot_nat_amend_stats[key] = max(tot_nat_amend_stats[key], nat_amend_stats[key])
        for key in tot_nat_amend_stats.keys():
            tot_nat_amend_stats[key] = max(tot_nat_amend_stats[key], nat_amend_stats[key])

        # Write data
        nat_amend_stats_df.reset_index(inplace=True)
        if i == 0:
            nat_str = ",".join(nat_amend_stats_df.columns)
            nat_str += "," + ",".join(nat_amend_stats.keys())
            files_dict["nat_amend"].write(nat_str + "\n")
        nat_str = df_to_str(nat_amend_stats_df)
        nat_str += "," + ",".join([str(el) for el in nat_amend_stats.values()])
        files_dict["nat_amend"].write(nat_str + "\n")

        if use_original:
            ########################
            # ORIGINAL NAT FORMULA #
            ########################
            orig_reg_bal_dfs, orig_reg_party_sizes = remove_inds(
                reg_bal_dfs=rand_gen.reg_bal_dfs,
                reg_party_sizes=rand_gen.reg_party_sizes
            )

            nat_formula_orig = NationalFormulaOriginal(reg_bal_dfs=orig_reg_bal_dfs,
                                                       reg_party_sizes=orig_reg_party_sizes,
                                                       comp_party_sizes=rand_gen.comp_party_sizes,
                                                       reg_tot_seats=rand_gen.reg_tot_seats)
            nat_formula_orig.calc_seats()

            # Stats
            nat_orig_stats_df, nat_orig_stats = nat_formula_orig.calc_nat_stats()
            for key in nat_orig_stats.keys():
                tot_nat_orig_stats[key] = max(tot_nat_orig_stats[key], nat_orig_stats[key])

            # Write data
            nat_orig_stats_df.reset_index(inplace=True)
            if i == 0:
                nat_str = ",".join(nat_orig_stats_df.columns)
                nat_str += "," + ",".join(nat_orig_stats.keys())
                files_dict["nat_orig"].write(nat_str + "\n")
            nat_str = df_to_str(nat_orig_stats_df)
            nat_str += "," + ",".join([str(el) for el in nat_orig_stats.values()])
            files_dict["nat_orig"].write(nat_str + "\n")

            ##################
            # DIFF NAT STATS #
            ##################
            nat_diff_stats_df = nat_orig_stats_df.copy()
            nat_diff_stats_df["votes"] -= nat_amend_stats_df["votes"]
            nat_diff_stats_df["seats"] -= nat_amend_stats_df["seats"]

            nat_diff_stats_df["perc_votes"] -= nat_amend_stats_df["perc_votes"]
            nat_diff_stats_df["perc_seats"] -= nat_amend_stats_df["perc_seats"]

            nat_diff_stats_df["diff_seats"] -= nat_amend_stats_df["diff_seats"]
            nat_diff_stats_df["diff_perc"] -= nat_amend_stats_df["diff_perc"]

            nat_diff_stats = {
                "max_diff_seats": nat_diff_stats_df["diff_seats"].max(),
                "mean_diff_seats": nat_diff_stats_df["diff_seats"].mean(),
                "max_diff_perc": nat_diff_stats_df["diff_perc"].max(),
                "mean_diff_perc": nat_diff_stats_df["diff_perc"].mean(),
            }

            for key in nat_diff_stats.keys():
                tot_nat_diff_stats[key] = max(tot_nat_diff_stats[key], nat_diff_stats[key])

            # Write data
            nat_diff_stats_df.reset_index(inplace=True)
            if i == 0:
                nat_str = ",".join(nat_diff_stats_df.columns)
                nat_str += "," + ",".join(nat_diff_stats.keys())
                files_dict["nat_diff"].write(nat_str + "\n")
            nat_str = df_to_str(nat_diff_stats_df)
            nat_str += "," + ",".join([str(el) for el in nat_diff_stats.values()])
            files_dict["nat_diff"].write(nat_str + "\n")

        # Progress bar stats
        if use_original:
            prog_stats = {
                "o_max": tot_nat_orig_stats["max_diff_perc"] * 100,
                "a_max": tot_nat_amend_stats["max_diff_perc"] * 100,
                "d_max": tot_nat_diff_stats["max_diff_perc"] * 100
            }
        else:
            prog_stats = {
                "o_max": tot_nat_orig_stats["max_diff_perc"] * 100
            }

        prog_bar.set_postfix(prog_stats)

        ####################
        # SAVE BALLOT DATA #
        ####################
        comp_bal_df = rand_gen.comp_bal_df
        comp_bal_df["party_size"] = rand_gen.comp_party_sizes
        comp_bal_df.reset_index(inplace=True)
        if i == 0:
            files_dict["comp"].write(",".join(comp_bal_df.columns) + "\n")
        files_dict["comp"].write(df_to_str(comp_bal_df) + "\n")

        for reg, reg_df in rand_gen.reg_bal_dfs.items():
            reg_df["party_size"] = rand_gen.reg_party_sizes[reg]
            reg_df.reset_index(inplace=True)
            if i == 0:
                files_dict[reg].write(",".join(reg_df.columns) + "\n")
            files_dict[reg].write(df_to_str(reg_df) + "\n")

    for file in files_dict.values():
        file.close()

