from electoral_formula.datagen.rand import RandomGenerator
from tqdm import tqdm
import pandas as pd


def main():
    data_gen_kwargs = {
        "pois_mean_n_parties": 40,
        "min_n_parties": 1,
        "mean_n_inds": 5,
        "min_n_inds": 1,
        "dist_type_n_inds": "pois",
        "n_inds_mode": "reg",
        "pois_mean_n_large_parties": 3,
        "min_n_large_parties": 0,
        "min_perc_large_votes": 0.01,
        "max_perc_large_votes": 0.99,
        "mean_perc_large_votes": 0.89,
        "std_perc_large_votes": 0.1,
        "dist_type_perc_large_votes": "trunc_normal",
        "mean_perc_party_votes": 0.85,
        "std_perc_party_votes": 0.1,
        "min_perc_party_votes": 0.01,
        "max_perc_party_votes": 0.99,
        "vary_voter_turnout": True,
        "mean_reg_voter_turnout": 0.66,
        "std_reg_voter_turnout": 0.1,
        "dist_type_voter_turnout": "trunc_normal",
        "vary_voter_turnout_tier": 1
    }

    # data_gen_kwargs = dict(
    #     pois_mean_n_parties=40, min_n_parties=1,
    #     mean_n_inds=5, min_n_inds=1, dist_type_n_inds="pois", n_inds_mode="reg",
    #     pois_mean_n_large_parties=3, min_n_large_parties=0,
    #     min_perc_large_votes=0.01, max_perc_large_votes=0.99,
    #     mean_perc_large_votes=0.89, std_perc_large_votes=0.1, dist_type_perc_large_votes="trunc_normal",
    #     mean_perc_party_votes=0.85, std_perc_party_votes=0.1,
    #     min_perc_party_votes=0.01, max_perc_party_votes=0.99,
    #     # voter_turnout
    #     vary_voter_turnout=True, mean_reg_voter_turnout=0.66,
    #     std_reg_voter_turnout=0.1, dist_type_voter_turnout="trunc_normal", vary_voter_turnout_tier=1
    # )

    rg = RandomGenerator(**data_gen_kwargs)

    # rg.save_reg_bal_dfs("data/ballots.csv")
    #
    # exit()

    # rg = RandomGenerator(min_n_parties=10, max_n_parties=60, norm_mean_n_parties=40, norm_std_n_parties=10, n_inds=10,
    #                      n_large_parties=5, perc_large_votes=1)
    # pois_mean_n_inds
    n_runs = 10000
    # d = {
    #     "n_parties": [],
    #     "n_inds": [],
    #     "n_large_parties": [],
    #     "perc_large_votes": [],
    #     "nat_voter_turnout": []
    # }

    for i in tqdm(range(n_runs)):
        rg = RandomGenerator(**data_gen_kwargs)
        # d["n_parties"].append(rg.n_parties)
        # d["act_n_inds"].append(rg.act_n_inds)
        # d["n_large_parties"].append(rg.n_large_parties)
        # d["perc_large_votes"].append(rg.perc_large_votes)
        # d["nat_voter_turnout"].append(rg.nat_voter_turnout)
        rg.save_reg_bal_dfs(f"data/ballot/ballots_{i}.csv")
        rg.save_gen_data(f"data/gen_data/gen_data_{i}.csv")
        rg.save_reg_data(f"data/reg_data/reg_data_{i}.csv")

    # df = pd.DataFrame.from_dict(d)
    # df.to_csv("data/data.csv", index=False)


if __name__ == "__main__":
    main()
