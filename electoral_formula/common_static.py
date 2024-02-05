# Common static variables
REGS = ['ec', 'fs', 'gp', 'kzn', 'lp', 'mp', 'nw', 'nc', 'wc']

REG_TOT_VOTES = {
    'ec': 2020527,
    'fs': 907212,
    'gp': 4537402,
    'kzn': 3652577,
    'lp': 1510568,
    'mp': 1271979,
    'nw': 994220,
    'nc': 410842,
    'wc': 2112170
}

REG_TOT_SEATS = {
    'ec': 25,
    'fs': 11,
    'gp': 48,
    'kzn': 41,
    'lp': 19,
    'mp': 15,
    'nw': 13,
    'nc': 5,
    'wc': 23
}

assert REG_TOT_SEATS.keys() == REG_TOT_VOTES.keys(), "Keys for REG_TOT_VOTES and REG_TOT_SEATS must be equal"

REGS = list(REG_TOT_SEATS.keys())