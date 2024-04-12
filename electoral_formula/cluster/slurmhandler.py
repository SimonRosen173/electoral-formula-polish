import json
import os
import subprocess

from electoral_formula.utils import join_paths, create_or_clear_folder


PATH_SEP = os.path.sep
BASE_PATH_ARR = os.path.abspath(__file__).split(PATH_SEP)[:-2]

BASE_PATH = os.path.normpath(PATH_SEP.join(BASE_PATH_ARR))
SLURMS_FOLDER = join_paths([BASE_PATH, "cluster", "slurms"])

API_PATH = join_paths([BASE_PATH, "api.py"])

SLURM_LOGS_FOLDER = "/datasets/srosen/logs/elec/slurms"
RUN_LOGS_FOLDER = "/datasets/srosen/logs/elec/runs"
DATA_FOLDER = "/datasets/srosen/logs/elec/data"

TMP_LOCAL_FOLDER = "/tmp/myrun"
TMP_LOCAL_DATA_FOLDER = "/tmp/myrun/data"


class SlurmHandler:
    def __init__(self, config_path, is_local=False):
        # Load config
        config_path = join_paths([BASE_PATH, "configs", config_path])
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = json.load(f)

        job_name = self.config["cluster"]["job_name"]

        self.run_logs_folder = join_paths([RUN_LOGS_FOLDER, job_name])

        self.cluster_data_path = join_paths([DATA_FOLDER, job_name])

        if not is_local:
            create_or_clear_folder(self.cluster_data_path)
            create_or_clear_folder(self.run_logs_folder)

    def create_slurm(self):
        template_path = os.path.join(BASE_PATH, "cluster", "template.slurm")
        with open(template_path, "r") as f:
            slurm_template = "".join(f.readlines())

        cluster_config = self.config["cluster"]
        slurm_str = slurm_template.replace("{partition}", cluster_config["partition"])
        slurm_str = slurm_str.replace("{job_name}", cluster_config["job_name"])
        slurm_str = slurm_str.replace("{slurm_logs_folder}", SLURM_LOGS_FOLDER)

        ##################
        # SETUP COMMANDS #
        ##################
        ...
        ##################

        ################
        # RUN COMMANDS #
        ################
        n_batches = self.config["exp"]["n_batches"]
        assert n_batches <= self.config["cluster"]["max_parallel_repeats"]
        run_commands = ""
        for curr_batch in range(n_batches):
            curr_run_command = f"python3 {API_PATH} --run {self.config_path} " \
                               f"--folder {TMP_LOCAL_DATA_FOLDER}/run_{curr_batch} -bn {curr_batch}"
            curr_run_command += f" > {self.run_logs_folder}/run_{curr_batch}.out 2>&1"
            curr_run_command += " &"
            run_commands += curr_run_command + "\n"
        slurm_str = slurm_str.replace("{run_commands}", run_commands)
        ################

        ###################
        # FINISH COMMANDS #
        ###################
        finish_commands = f"python3 {API_PATH} --join {TMP_LOCAL_DATA_FOLDER} --n-batches {n_batches} --formulae amend_orig"

        # zip files
        # finish_commands += "\n" + f"zip -r {TMP_LOCAL_FOLDER}/data.zip {TMP_LOCAL_DATA_FOLDER}/agr/*"
        finish_commands += "\n" + f"zip -r {TMP_LOCAL_FOLDER}/data.zip {TMP_LOCAL_DATA_FOLDER}/agr/*"
        # finish_commands += "\n" + f"zip -r {TMP_LOCAL_FOLDER}/data.zip {TMP_LOCAL_DATA_FOLDER}/*"
        finish_commands += "\n" + f"mv {TMP_LOCAL_FOLDER}/data.zip {self.cluster_data_path}/data.zip"
        slurm_str = slurm_str.replace("{finish_commands}", finish_commands)
        ###################

        slurm_path = join_paths([SLURMS_FOLDER, "job.slurm"])
        with open(slurm_path, "w") as f:
            f.write(slurm_str)


if __name__ == "__main__":
    sh = SlurmHandler("exp1.json", is_local=True)
    sh.create_slurm()
