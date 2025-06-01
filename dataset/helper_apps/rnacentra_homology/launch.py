import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="", type=str)
args = parser.parse_args()

db_cmd = f"./run.sh {args.input}"

subprocess.run(db_cmd, shell=True, executable="/bin/bash")

