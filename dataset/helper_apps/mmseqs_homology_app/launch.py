import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--range", default="", type=str)
args = parser.parse_args()

db_cmd = "./create_mmseqs_db.sh"

if args.range != "":
    search_cmd = f"./inter_species_search_mmseqs.sh {args.range}"
else:
    search_cmd = f"./inter_species_search_mmseqs.sh"

subprocess.run(db_cmd, shell=True, executable="/bin/bash")
subprocess.run(search_cmd, shell=True, executable="/bin/bash")