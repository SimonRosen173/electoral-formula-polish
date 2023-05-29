# Electoral Formula

## Requirements
The packages required to run this code are found in the 'requirements.txt'

## Command Line Interface
Two command line interfaces (CLIs) exist:
1. **Electoral Formula CLI:** This provides an interface for calculating 
the seats awarded given the regional and compensatory ballot for both 
the original and amended electoral formulae.
2. **Benchmarker CLI:** This provides an interface for running the 
experiments described in the report.

The CLI exist in the folder electoral_formula/scripts

For both CLIs, use the `-h` or `--help` flags to show all the available flags and their related info.

### Electoral Formula CLI
**Path:** electoral_formula/scripts/formula.py

Using this CLI one can provide the paths to the regional ballots, 
compensatory ballot & sizes of party lists and the program will create 
a file with the calculated seats and relevant stats using the specified 
electoral formula. 

This CLI uses a variety of flags which are given below:

#### Required Flags
- `--alg`, `-alg` -> Either 'orig' or 'amend' and specifies which electoral
formula is being used. If 'orig' is selected then independents in the ballots
will not be considered.
- `--reg-bal-path`,`-rbp` -> Path of excel doc containing votes and 
party list sizes for regional ballot across all regions. This doc must 
follow the format of electoral_formula/data/ballots/reg_template.xlsx. The path must 
either be absolute or relative to formula.py.
- `--comp-bal-path`,`-cbp` -> Path of excel doc containing votes and 
party list sizes for compensatory ballot. This doc must follow the format of 
electoral_formula/data/ballots/comp_template.xlsx. This arg is also required
when computing the seats for the original formula, as it needs the size 
of the party lists for the comp ballot. In the case of the original formula 
any vote values can be given. 
- `--seats-path`, `-sp` -> Path of new excel file to be created which will store 
the calculated seat allocations.
- `--stats-path`, `-stp` -> Path of new excel file to be created which will store 
stats associated with the seat calculations.

These flags can also be found using `-h` or `--help`.

#### Usage

Assuming that the current working directory is in the scripts folder.
```
python formula.py --alg <orig/amend> --reg-bal-path <path> --comp-bal-path <path> --seats-path <path> --stats-path <path>
``` 

#### Example Usage
Assuming that the current working directory is in the scripts folder.
```
python formula.py --alg orig --reg-bal-path data/ballots/reg_2019.xlsx --comp-bal-path data/ballots/comp_2019.xlsx --seats-path data/results/seats.xlsx --stats-path data/results/stats.xlsx
``` 
Note: The linebreak was introduced for formatting and should not be included in the terminal.

#### Format of Ballots
The excel docs provided as input must strictly follow the format of the templates provided and all values must be 
specified. If you wish to simulate a party or independent not running in a specific region then you must simply 
set the votes won for that region to zero and not leave the cell empty.  

## Benchmarker CLI
**Path:** electoral_formula/scripts/benchmarker.py

This command line interface allows for a user to run the experiments mentioned in the report. There are two types of 
experiments that can be run:
1. Random
    - n simulations are run with ballot data being created randomly according to specified parameters. 
2. Increasing votes
    - The votes of the first party is increased for a varying range of data

Further information about these experiments can be found in the report.
### Required Flags
- `-e`, `--exp`: Which experiment to run. Valid values = 'rand' or 'incr_votes'.
- `-f`, `--folder`: Folder in which results will be saved.
- `-nr`, `--n_runs`: Number of simulations to be run
- `-fo`, `--formula`: Which formula or formulae to use. When exp is 'incr_votes' then 
the valid values of formula is 'orig', 'amend' and 'amend_orig'. When exp is 'rand' then 
the valid values of formula is 'amend' and 'amend_orig'.
- `-cbr`, `--comp-bal-from-reg`: If true, compute compensatory ballot from regional otherwise comp ballot is random.
- `-np`, `--n-parties`: Number of parties
- `-ni`, `--n-inds`: Number of independents
- `-nlp`, `--n-large-parties`: Number of 'larger' parties
- `-ppv`, `--perc-party-vote`: Percentage of votes reserved for parties
- `-plmin`, `--perc-large-votes-min`: Minimum percentage of votes reservable for large parties.
- `-plmax`, `--perc-large-votes-max`: Maximum percentage of votes reservable for large parties.

#### Required flags if exp is incr_votes
The following flags are only required and applicable when exp='incr_votes'.
- `-rpv`, `--repeats_per_vote`: Number of repeated runs per specified votes.
- `-sp`, `--start_perc`: Start percentage of votes assigned to party 1.
- `-ep`, `--end_perc`: End percentage of votes assigned to party 1.