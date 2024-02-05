import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import random
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np

from scipy.stats import truncnorm

from electoral_formula.common_static import REG_TOT_VOTES, REG_TOT_SEATS, REGS


# Generate n random votes with condition that sum of votes must equal tot_votes
# Distribution of random votes is provably uniform random
#   Source: Sampling Uniformly from the Unit Simplex (Noah A. Smith and Roy W. Tromble)
def gen_rand_partitions(tot_votes, n, max_tries=1000, allow_zeros=True):
    is_repeats = True
    tmp_arr = None

    if not allow_zeros:
        if n > tot_votes:
            raise ValueError(f"n cannot be greater than tot_votes when allow_zeros=False. n={n}, tot_votes={tot_votes}")
        curr_try = 0
        while is_repeats:
            tmp_arr = np.random.randint(0, tot_votes, n - 1)
            is_repeats = len(np.unique(tmp_arr)) != len(tmp_arr)
            tmp_arr.sort()
            curr_try += 1
            if curr_try > max_tries:
                raise ValueError(f"max_tries={max_tries} reached for tot_votes={tot_votes}, n={n}")
    else:
        tmp_arr = np.random.randint(0, tot_votes, n - 1)
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

        if n_large > 0:
            large_votes = rand_votes_fn(tot_large_votes, n_large)
        else:
            large_votes = []

        if n_small > 0:
            if tot_small_votes < n_small:
                # If there are not enough votes to divide
                n_small_zero = n_small - tot_small_votes
                zero_votes = [0 for _ in range(n_small_zero)]
                if n_small - n_small_zero > 0:
                    small_votes = rand_votes_fn(n_small - tot_small_votes, n_small - n_small_zero)
                    small_votes.extend(zero_votes)
                else:
                    small_votes = zero_votes
                random.shuffle(small_votes)
            else:
                small_votes = rand_votes_fn(tot_small_votes, n_small)
        else:
            small_votes = []

        votes = list(np.concatenate([votes, large_votes, small_votes]))
    else:
        votes = list(np.concatenate([votes, rand_votes_fn(tot_votes, n_parties)]))

    return votes


def trunc_norm(min_val, max_val, mean, std, n, round_to_int=False):
    a, b = (min_val - mean) / std, (max_val - mean) / std
    vals = truncnorm.rvs(a, b, loc=mean, scale=std, size=n)
    if round_to_int:
        vals = np.round(vals).astype(int)
    if len(vals) == 1:
        return vals[0]
    else:
        return vals


def bounded_poisson(
        min_val: int,
        mean: int
):
    val = np.random.poisson(mean - min_val, size=1)[0] + min_val
    return val


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

            # voter_turnout #  # TODO
            voter_turnout: Optional[float] = None,
            vary_voter_turnout_mode: str = "const",
            # `const` - constant across all regions, `reg` - varied across regions
            min_voter_turnout: Optional[float] = None,
            mean_voter_turnout: Optional[float] = None,
            std_voter_turnout: Optional[float] = None,
            max_voter_turnout: Optional[float] = None,
            voter_turnout_dist: Optional[str] = None,  # ['trunc_normal', 'uniform']

            # n_parties #
            n_parties: Optional[int] = None,
            min_n_parties: Optional[int] = 1,
            max_n_parties: Optional[int] = None,
            norm_mean_n_parties: Optional[int] = None,  # Normal distribution of n_parties around 'norm_mean_n_parties'
            norm_std_n_parties: Optional[int] = None,
            pois_mean_n_parties: Optional[int] = None,  # Poisson distribution

            # n_large_parties #
            n_large_parties: Optional[int] = None,
            min_n_large_parties: Optional[int] = 0,
            max_n_large_parties: Optional[int] = None,
            norm_mean_n_large_parties: Optional[int] = None,  # Normal distribution of n_large_parties around 'norm_mean_n_large_parties'
            norm_std_n_large_parties: Optional[int] = None,
            pois_mean_n_large_parties: Optional[int] = None,  # Poisson distribution

            # perc_large_votes #
            perc_large_votes: Optional[float] = None,
            min_perc_large_votes: Optional[float] = None,
            max_perc_large_votes: Optional[float] = None,
            norm_mean_perc_large_votes: Optional[float] = None,  # Normal distribution of perc_large_votes around 'norm_mean_perc_large_votes'
            norm_std_perc_large_votes: Optional[float] = None,

            # n_inds #
            n_inds: Optional[int] = None,  # Number of independents
            min_n_inds: Optional[int] = 1,
            max_n_inds: Optional[int] = None,
            norm_mean_n_inds: Optional[int] = None,  # Normal distribution of n_ind around 'norm_mean_n_inds'
            norm_std_n_inds: Optional[int] = None,
            pois_mean_n_inds: Optional[int] = None,  # Poisson distribution

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
        :param min_perc_large_votes: If perc_large_votes is randomly sampled this is the minimum value
        :param max_perc_large_votes: If perc_large_votes is randomly sampled this is the maximum value
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

        if reg_tot_seats is None:
            reg_tot_seats = REG_TOT_SEATS
        self.reg_tot_seats = reg_tot_seats

        ######################
        # VARY VOTER TURNOUT #
        #####################
        if reg_tot_votes is None:
            reg_tot_votes = REG_TOT_VOTES

        self.is_vary_voter_turnout = (voter_turnout is not None) or (min_voter_turnout is not None)

        self.reg_tot_votes = reg_tot_votes

        ###########################################
        # PERCENTAGE OF VOTES ASSIGNED TO PARTIES #
        ###########################################
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
        ###########################################

        if comp_tot_votes is None:
            comp_tot_votes = sum(reg_tot_votes.values())
        self.comp_tot_votes = comp_tot_votes
        self.comp_tot_seats = comp_tot_seats

        ################
        #   PARTIES    #
        ################
        if n_parties is None:
            if min_n_parties < 1:
                raise ValueError(f"min_n_parties must be greater than 1. (min_n_parties = {min_n_parties})")

            # Truncated Normal Distribution
            if norm_mean_n_parties is not None and norm_std_n_parties is not None:
                if min_n_parties is None or max_n_parties is None:
                    raise ValueError("If n_parties is none, then min_n_parties and max_n_parties must be set")

                # Calculate n_parties using truncated normal distribution
                n_parties = trunc_norm(min_n_parties, max_n_parties,
                                       norm_mean_n_parties, norm_std_n_parties,
                                       n=1, round_to_int=True)
            elif norm_mean_n_parties is not None or norm_std_n_parties is not None:
                raise ValueError("Either norm_mean_n_parties and norm_std_n_parties must be set, not just one.")
            # Poisson Distribution
            elif pois_mean_n_parties is not None:
                n_parties = np.random.poisson(lam=pois_mean_n_parties, size=1)[0]
            else:
                if min_n_parties is None or max_n_parties is None:
                    raise ValueError("If n_parties is none, then min_n_parties and max_n_parties must be set")

                # Calculate n_parties using uniform random distribution
                n_parties = np.random.randint(min_n_parties, max_n_parties + 1)

        self.n_parties = n_parties
        ################

        #################
        # N LARGE PARTIES #
        #################
        large_parties_vals = [min_n_large_parties, max_n_large_parties, norm_mean_n_large_parties, norm_std_n_large_parties, pois_mean_n_large_parties]
        self.is_large_parties = not np.all([x is None for x in large_parties_vals])
        if n_large_parties is not None and n_large_parties == 0:
            self.is_large_parties = False
        # if n_large_parties is None and \
        #         min_n_large_parties is None and max_n_large_parties is None and \
        #         norm_mean_n_large_parties is None and norm_std_n_large_parties is None:
        #
        #     self.is_large_parties = False
        if self.is_large_parties:
            # self.is_large_parties = True

            if n_large_parties is not None:
                assert n_large_parties > 0
                # Do something?
                pass
            # Normal Distribution
            elif norm_mean_n_large_parties is not None and norm_std_n_large_parties is not None:
                assert min_n_large_parties is not None and max_n_large_parties is not None

                # Ensure n_large_parties is not bigger than n_parties
                max_n_large_parties = min(max_n_large_parties, n_parties)

                # Calculate n_parties using truncated normal distribution
                n_large_parties = trunc_norm(min_n_large_parties, max_n_large_parties,
                                             norm_mean_n_large_parties, norm_std_n_large_parties,
                                             n=1, round_to_int=True)
                pass
            # Poisson Distribution
            elif pois_mean_n_large_parties is not None:
                n_large_parties = bounded_poisson(min_n_large_parties, pois_mean_n_large_parties)
                # n_large_parties = np.random.poisson(lam=pois_mean_n_large_parties, size=1)[0]
            # Uniform Random
            elif min_n_large_parties is not None and max_n_large_parties is not None:
                # Ensure n_large_parties is not bigger than n_parties
                max_n_large_parties = min(max_n_large_parties, n_parties)

                # Uniform random
                n_large_parties = np.random.randint(min_n_large_parties, max_n_large_parties + 1)
            else:
                raise ValueError
        self.n_large_parties = n_large_parties

        if n_large_parties == 0:
            self.is_large_parties = False
        # if n_large_parties is None or n_large_parties <= 0:
        #     self.is_large_parties = False
        # else:
        #     self.is_large_parties = True

        # PERC LARGE VOTES #
        # if perc_large_votes is None and \
        #         min_perc_large_votes is None and max_perc_large_votes is None and \
        #         norm_mean_perc_large_votes is None and norm_std_perc_large_votes is None:
        #     # Do something?
        #     pass
        # else:
        if self.is_large_parties:

            if min_perc_large_votes is None:
                min_perc_large_votes = 0
            if max_perc_large_votes is None:
                max_perc_large_votes = 1
            assert 0 <= max_perc_large_votes <= 1

            if perc_large_votes is not None:
                assert 0 <= perc_large_votes <= 1
                # Do something?
                pass
            # Truncated Normal Distribution
            elif norm_mean_perc_large_votes is not None and norm_std_perc_large_votes is not None:
                # assert min_perc_large_votes is not None and max_perc_large_votes is not None

                # Ensure perc_large_votes is not bigger than n_parties
                max_perc_large_votes = min(max_perc_large_votes, n_parties)

                # Calculate n_parties using truncated normal distribution
                perc_large_votes = trunc_norm(min_perc_large_votes, max_perc_large_votes,
                                              norm_mean_perc_large_votes, norm_std_perc_large_votes,
                                              n=1, round_to_int=False)
            elif min_perc_large_votes is not None and max_perc_large_votes is not None:
                # # Ensure perc_large_votes is not bigger than n_parties
                # max_perc_large_votes = min(max_perc_large_votes, n_parties)

                # Uniform random
                perc_large_votes = np.random.uniform(min_perc_large_votes, max_perc_large_votes)
            else:
                raise ValueError

            assert min_perc_large_votes <= perc_large_votes <= max_perc_large_votes

        self.perc_large_votes = perc_large_votes
        # if perc_large_votes is None and self.is_large_parties:
        #     perc_large_votes = np.random.uniform(perc_large_votes_min, perc_large_votes_max)
        #
        # self.n_large_parties = n_large_parties
        # if self.is_large_parties and perc_large_votes_min >= perc_large_votes_max:
        #     raise ValueError("Condition 'perc_large_votes_min < perc_large_votes_max' not met")

        # self.perc_large_votes = perc_large_votes
        # self.perc_large_votes_min = min_perc_large_votes
        # self.perc_large_votes_max = max_perc_large_votes
        ################

        ################
        # INDEPENDENTS #
        ################
        if n_inds is None:
            if min_n_inds < 1:
                raise ValueError(f"min_n_inds cannot be less than 1. (min_n_inds={min_n_inds})")

            # Normal Distribution
            if norm_mean_n_inds is not None and norm_std_n_inds is not None:
                if min_n_inds is None or max_n_inds is None:
                    raise ValueError("If n_inds is none, then min_n_inds and max_n_inds must be set")
                # Calculate n_inds using truncated normal distribution
                n_inds = trunc_norm(min_n_inds, max_n_inds,
                                    norm_mean_n_large_parties, norm_std_n_large_parties,
                                    n=1, round_to_int=True)
            elif norm_mean_n_inds is not None or norm_std_n_inds is not None:
                raise ValueError("Either norm_mean_n_inds and norm_std_n_inds must be set, not just one.")
            # Poisson Distribution
            elif pois_mean_n_inds is not None:
                n_inds = bounded_poisson(min_n_inds, pois_mean_n_inds)
                # n_inds = np.random.poisson(lam=pois_mean_n_inds, size=1)[0]
            # Uniform random
            else:
                if min_n_inds is None or max_n_inds is None:
                    raise ValueError("If n_inds is none, then min_n_inds and max_n_inds must be set")
                n_inds = np.random.randint(min_n_inds, max_n_inds + 1)
        self.n_inds = n_inds
        ################

        ##################
        # Error checking #
        ##################
        if self.n_inds < min_n_inds:
            raise ValueError(f"n_inds={self.n_inds} < min_n_inds={min_n_inds}")
        if self.n_parties < min_n_parties:
            raise ValueError(f"n_parties={n_parties} < min_n_parties={min_n_parties}")
        if self.n_large_parties > n_parties:
            raise ValueError(f"n_large_parties={self.n_large_parties} > n_parties={n_parties}")
        ##################

        ######################
        # GEN PARTY VOTES FN #
        ######################
        def gen_party_votes_fn(tot_votes):
            votes = gen_party_votes(
                tot_votes=tot_votes,
                n_parties=self.n_parties,
                is_large_votes=self.is_large_parties,
                rand_votes_fn=gen_rand_votes_fn,
                perc_large_votes=perc_large_votes,
                n_large=n_large_parties,
                first_party_vote_perc=first_party_vote_perc
            )
            return votes

        self.gen_party_votes_fn = gen_party_votes_fn
        ######################

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

    # Amalgamate all regional ballots into one df and save to csv
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


