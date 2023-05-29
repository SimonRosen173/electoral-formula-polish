
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np

from electoral_formula.common_static import REG_TOT_VOTES, REG_TOT_SEATS


# Generate n random votes with condition that sum of votes must equal tot_votes
# Distribution of random votes is provably uniform random
#   Source: Sampling Uniformly from the Unit Simplex (Noah A. Smith and Roy W. Tromble)
def gen_rand_partitions(tot_votes, n):
    is_repeats = True
    tmp_arr = None
    while is_repeats:
        tmp_arr = np.random.randint(0, tot_votes, n - 1)
        is_repeats = len(np.unique(tmp_arr)) != len(tmp_arr)
        tmp_arr.sort()

    tmp_arr = np.insert(tmp_arr, 0, 0)
    tmp_arr = np.append(tmp_arr, tot_votes)

    votes = []
    for i in range(len(tmp_arr) - 1):
        votes.append(tmp_arr[i + 1] - tmp_arr[i])

    return votes


def gen_party_votes(
        tot_votes,
        n_parties,
        rand_votes_fn,
        is_large_votes,
        perc_large_votes=None,
        n_large=None,
        first_party_vote_perc=None
):
    if first_party_vote_perc is not None:
        votes = [int(first_party_vote_perc*tot_votes)]
        tot_votes = tot_votes - votes[0]
        n_parties = n_parties - 1
    else:
        votes = []

    if is_large_votes:
        if perc_large_votes is None or n_large is None:
            raise ValueError("When is_large_votes=True then perc_large_votes and n_large must be set")
        tot_large_votes = int(tot_votes * perc_large_votes)
        tot_small_votes = tot_votes - tot_large_votes
        n_small = n_parties - n_large

        large_votes = rand_votes_fn(tot_large_votes, n_large)
        small_votes = rand_votes_fn(tot_small_votes, n_small)

        votes = list(np.concatenate([votes, large_votes, small_votes]))
    else:
        votes = list(np.concatenate([votes, rand_votes_fn(tot_votes, n_parties)]))

    return votes


#########################################
#           RandomGenerator             #
# Class used to generate data in report #
#########################################
class RandomGenerator:
    # TODO: Add docs
    def __init__(
            self,
            reg_tot_votes: Dict[str, int] = None,
            reg_tot_seats: Dict[str, int] = None,
            comp_tot_votes: int = None,
            comp_tot_seats: int = 200,

            n_parties: Optional[int] = None,
            min_n_parties: Optional[int] = None,
            max_n_parties: Optional[int] = None,

            n_large_parties: Optional[int] = None,

            perc_large_votes: Optional[float] = None,
            perc_large_votes_min: Optional[float] = None,
            perc_large_votes_max: Optional[float] = None,

            n_inds: Optional[int] = None,  # Number of independents
            min_n_inds: Optional[int] = None,
            max_n_inds: Optional[int] = None,

            first_party_vote_perc=None,

            perc_reg_parties_votes: Optional[Union[Dict[str, float], float]] = None,
            inflate_votes=False,
            comp_bal_from_reg=True,  # If comp_bal is based off regional values
            gen_rand_votes_fn=gen_rand_partitions,
    ):
        """
        Generates random regional ballot & compensatory ballot data according to specified parameters
        :param reg_tot_votes: Dictionary of total valid votes cast for each region
        :type reg_tot_votes: Dict[str, int]
        :param reg_tot_seats: Dictionary of total regional seats available for each region
        :type reg_tot_seats: Dict[str, int]
        :param comp_tot_votes: Total valid votes cast on compensatory ballot
        :param comp_tot_seats: Total compensatory seats available
        :param n_parties: Number of parties. Can be equal to zero, where there is no distinction between 'large'
        and 'small' parties. If not set then n_parties is randomly sampled from [min_n_parties, max_n_parties].
        :param min_n_parties: When randomly sampling n_parties this is the minimum no of parties
        :param max_n_parties: When randomly sampling n_parties this is the maximum no of parties
        :param n_large_parties: No of "large parties" as described in the report
        :param perc_large_votes: Percentage of total votes assigned to 'large parties'. If not set then
        this is randomly sampled from [perc_large_votes_min, perc_large_votes_max]
        :param perc_large_votes_min: If perc_large_votes is randomly sampled this is the minimum value
        :param perc_large_votes_max: If perc_large_votes is randomly sampled this is the maximum value
        :param n_inds: Number of independents. If not set then n_inds is randomly sampled from
        [min_n_inds, max_n_inds]
        :param min_n_inds: If n_inds is randomly sampled then this is the minimum value
        :param max_n_inds: If n_inds is randomly sampled then this is the minimum value
        :param first_party_vote_perc: If set the votes of the first party is set based off this percentage. If not
        set then votes of first party is random according to other parameters.
        :param perc_reg_parties_votes: Percentage of votes assigned to parties
        :param inflate_votes: If True and comp_bal_from_reg is True then this 'inflates' the compensatory votes so that
        the distributions of votes is still the same as the regional ballot and the total votes is the same as the
        regional ballot. The votes must be inflated to account for the votes taken by the independents.
        :param comp_bal_from_reg: If True then comp ballot is set to be the same as the total votes per party of the
        regional ballot
        :param gen_rand_votes_fn: Function that is used to calculate random votes
        """
        self.gen_rand_votes_fn = gen_rand_votes_fn
        self.is_large_parties = n_large_parties > 0

        if perc_large_votes is None and self.is_large_parties:
            perc_large_votes = np.random.uniform(perc_large_votes_min, perc_large_votes_max)

        def gen_party_votes_fn(tot_votes):
            votes = gen_party_votes(
                tot_votes=tot_votes,
                n_parties=n_parties,
                is_large_votes=self.is_large_parties,
                rand_votes_fn=gen_rand_votes_fn,
                perc_large_votes=perc_large_votes,
                n_large=n_large_parties,
                first_party_vote_perc=first_party_vote_perc
            )
            return votes

        self.gen_party_votes_fn = gen_party_votes_fn

        if reg_tot_votes is None:
            reg_tot_votes = REG_TOT_VOTES
        self.reg_tot_votes = reg_tot_votes

        if reg_tot_seats is None:
            reg_tot_seats = REG_TOT_SEATS
        self.reg_tot_seats = reg_tot_seats

        if perc_reg_parties_votes is None:
            perc_reg_parties_votes = {}
            for reg in reg_tot_votes.keys():
                perc_reg_parties_votes[reg] = np.random.rand()
        elif type(perc_reg_parties_votes) == float:
            tmp_perc = perc_reg_parties_votes
            perc_reg_parties_votes = {}
            for reg in reg_tot_votes.keys():
                perc_reg_parties_votes[reg] = tmp_perc

        self.perc_reg_parties_votes = perc_reg_parties_votes

        if comp_tot_votes is None:
            comp_tot_votes = sum(reg_tot_votes.values())
        self.comp_tot_votes = comp_tot_votes
        self.comp_tot_seats = comp_tot_seats

        # PARTIES
        if n_parties is None:
            if min_n_parties is None or max_n_parties is None:
                raise ValueError("If n_parties is none, then min_n_parties and max_n_parties must be set")
            n_parties = np.random.randint(min_n_parties, max_n_parties + 1)
        self.n_parties = n_parties

        self.n_large_parties = n_large_parties
        if self.is_large_parties and perc_large_votes_min >= perc_large_votes_max:
            raise ValueError("Condition 'perc_large_votes_min < perc_large_votes_max' not met")

        self.perc_large_votes = perc_large_votes
        self.perc_large_votes_min = perc_large_votes_min
        self.perc_large_votes_max = perc_large_votes_max

        # INDEPENDENTS
        if n_inds is None:
            if min_n_inds is None or max_n_inds is None:
                raise ValueError("If n_inds is none, then min_n_inds and max_n_inds must be set")
            n_inds = np.random.randint(min_n_inds, max_n_inds + 1)
        self.n_inds = n_inds

        # BALLOTS
        self.reg_bal_dfs = None
        self.comp_bal_df = None
        self.agr_reg_bal_df = None

        # SIZES
        self.reg_party_sizes = None
        self.comp_party_sizes = None

        # INITS
        self._gen_reg_bal_dfs()
        self._create_agr_reg_bal_df()
        self._gen_comp_bal_df(inflate_votes=inflate_votes, from_reg=comp_bal_from_reg)
        self._gen_party_sizes()

    def _gen_reg_bal_dfs(self):
        reg_bal_dfs = {}
        for reg, tot_votes in self.reg_tot_votes.items():
            tot_party_votes = int(self.perc_reg_parties_votes[reg] * tot_votes)
            tot_ind_votes = tot_votes - tot_party_votes
            n_parties = self.n_parties
            n_inds = self.n_inds

            party_votes = self.gen_party_votes_fn(tot_party_votes)
            ind_votes = self.gen_rand_votes_fn(tot_ind_votes, n_inds)

            reg_df_dict = {
                "party": [f"party_{i+1}" for i in range(n_parties)] + [f"ind_{i+1}" for i in range(n_inds)],
                "votes": party_votes + ind_votes,
                "is_ind": [False for _ in range(n_parties)] + [True for _ in range(n_inds)]
            }
            reg_df = pd.DataFrame.from_dict(reg_df_dict)
            reg_df.set_index("party", inplace=True)
            reg_df["votes"] = reg_df["votes"].astype(int)
            reg_bal_dfs[reg] = reg_df

        self.reg_bal_dfs = reg_bal_dfs

    def _create_agr_reg_bal_df(self):
        first_reg = list(self.reg_bal_dfs.keys())[0]
        self.agr_reg_bal_df = self.reg_bal_dfs[first_reg].copy()
        for reg, df in self.reg_bal_dfs.items():
            if reg != first_reg:
                self.agr_reg_bal_df["votes"] += df["votes"]

    def _gen_comp_bal_df(self, inflate_votes=False, from_reg=True):
        tot_votes = self.comp_tot_votes
        n_parties = self.n_parties

        if from_reg:
            if inflate_votes:
                raise NotImplementedError

            tmp = self.agr_reg_bal_df[self.agr_reg_bal_df["is_ind"]==False]
            comp_bal_df = tmp[["votes"]].copy()

        else:
            party_votes = self.gen_party_votes_fn(tot_votes)

            comp_bal_dict = {
                "party": [f"party_{i+1}" for i in range(n_parties)],
                "votes": party_votes,
            }
            comp_bal_df = pd.DataFrame.from_dict(comp_bal_dict)
            comp_bal_df.set_index("party", inplace=True)

        comp_bal_df["votes"] = comp_bal_df["votes"].astype(int)
        self.comp_bal_df = comp_bal_df

    def _gen_party_sizes(self):
        n_parties = self.n_parties
        n_inds = self.n_inds

        # Regional
        reg_bal_dfs = self.reg_bal_dfs
        assert reg_bal_dfs is not None
        reg_party_sizes = {}
        for reg, tot_seats in self.reg_tot_seats.items():
            curr_party_sizes = [tot_seats for _ in range(n_parties)] + [1 for _ in range(n_inds)]
            parties = list(reg_bal_dfs[reg].index)
            df_dict = {
                "party": parties,
                "party_size": curr_party_sizes
            }
            df = pd.DataFrame.from_dict(df_dict)
            reg_party_sizes[reg] = df.set_index("party")["party_size"]
        self.reg_party_sizes = reg_party_sizes

        # Compensatory
        comp_bal_df = self.comp_bal_df
        assert comp_bal_df is not None
        df_dict = {
            "party": list(comp_bal_df.index),
            "party_size": [self.comp_tot_seats for _ in range(n_parties)]
        }
        df = pd.DataFrame.from_dict(df_dict)
        self.comp_party_sizes = df.set_index("party")["party_size"]

    # Amalgamate into one df and save to csv
    def save_reg_bal_dfs(self, file_path):
        reg_bal_dfs = self.reg_bal_dfs
        first_region = list(reg_bal_dfs.keys())[0]
        amalg_df = reg_bal_dfs[first_region].copy().reset_index()
        amalg_df["region"] = first_region

        for reg, reg_df in reg_bal_dfs.items():
            if reg != first_region:
                tmp_df = reg_df.copy().reset_index()
                tmp_df["region"] = reg
                amalg_df = pd.concat([amalg_df, tmp_df], ignore_index=True)

        amalg_df.to_csv(file_path, index=False)


