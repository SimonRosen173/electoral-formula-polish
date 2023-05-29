# Remove independents
def remove_inds(
        reg_bal_dfs,
        reg_party_sizes
):
    rem_reg_bal_dfs = {}
    rem_reg_party_sizes = {}
    for reg, reg_df in reg_bal_dfs.items():
        inds = reg_df[reg_df["is_ind"]==True].index
        reg_df = reg_df.drop(inds)
        reg_df = reg_df.drop("is_ind", axis=1)
        curr_party_sizes = reg_party_sizes[reg].drop(inds)
        rem_reg_bal_dfs[reg] = reg_df
        rem_reg_party_sizes[reg] = curr_party_sizes

    return rem_reg_bal_dfs, rem_reg_party_sizes


def df_to_str(df):
    df_arr = ["|".join(df[col].astype(str).tolist()) for col in df.columns]
    df_str = ",".join(df_arr)
    return df_str
