from electoral_formula.datagen.rand import RandomGenerator
from tqdm import tqdm
import pandas as pd


def main():
    # rg = RandomGenerator(min_n_parties=10, max_n_parties=60, norm_mean_n_parties=40, norm_std_n_parties=10, n_inds=10,
    #                      n_large_parties=5, perc_large_votes=1)
    # pois_mean_n_inds
    n_runs = 25000
    d = {
        "n_parties": [],
        "n_inds": [],
        "n_large_parties": [],
        "perc_large_votes": []
    }

    for i in tqdm(range(n_runs)):
        rg = RandomGenerator(pois_mean_n_parties=40, min_n_parties=1,
                             pois_mean_n_inds=5, min_n_inds=1,
                             pois_mean_n_large_parties=3,
                             min_perc_large_votes=0.01, max_perc_large_votes=0.99,
                             norm_mean_perc_large_votes=0.89, norm_std_perc_large_votes=0.1)
        d["n_parties"].append(rg.n_parties)
        d["n_inds"].append(rg.n_inds)
        d["n_large_parties"].append(rg.n_large_parties)
        d["perc_large_votes"].append(rg.perc_large_votes)
        # rg.save_reg_bal_dfs("data/ballots.csv")

    df = pd.DataFrame.from_dict(d)
    df.to_csv("data.csv", index=False)


if __name__ == "__main__":
    main()
