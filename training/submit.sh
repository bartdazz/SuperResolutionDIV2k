#!/bin/bash
# Usage: bash submit.sh <training_script.py> ["run description"]
# e.g.:  bash submit.sh train_cond_sr.py "Testing epsilon=0.01, all else baseline"
#
# Submits a SLURM job. The timestamp is generated here so that runs/ and
# slurm_logs/ directories share the exact same name for easy pairing.

SCRIPT=${1:?"Usage: bash submit.sh <training_script.py> [description]"}
DESCRIPTION=${2:-""}

VARIANT=$(python3 -c "
import ast, sys
src = open('$SCRIPT').read()
for n in ast.walk(ast.parse(src)):
    if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'VARIANT' for t in n.targets
    ):
        print(n.value.value); sys.exit()
")

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
DATE_PART=${TIMESTAMP%%_*}
TIME_PART=${TIMESTAMP#*_}
LOG_DIR="../slurm_logs/${DATE_PART}/${TIME_PART}_${VARIANT}"
mkdir -p "$LOG_DIR"
# Resolve to absolute path after ensuring directory exists
LOG_DIR="$(cd "$LOG_DIR" && pwd)"


sbatch <<EOF
#!/bin/bash -l
#SBATCH --job-name=${VARIANT}
#SBATCH --output=${LOG_DIR}/slurm_%j.out
#SBATCH --error=${LOG_DIR}/slurm_%j.err
#SBATCH --partition=testing
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00
#SBATCH --exclude=vgpu11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=bartolo.dazzini@studenti.unipd.it

### Some useful informative commands
echo -n 'Job started at: '
TZ="Europe/Rome" date
echo -n 'Directory: '
pwd
echo -n 'This job will be executed on th following nodes: '
echo \${SLURM_NODELIST}
echo


export RUN_TIMESTAMP="${TIMESTAMP}"
export RUN_DESCRIPTION="${DESCRIPTION}"
cd $(pwd)
conda activate gpuenv
export LD_PRELOAD=/home/bdazzini/.conda/envs/gpuenv/lib/libstdc++.so.6
/home/bdazzini/.conda/envs/gpuenv/bin/python -u ${SCRIPT}
echo -n 'Job finished at: '
TZ="Europe/Rome" date
EOF

echo "Submitted ${SCRIPT} (variant=${VARIANT}, timestamp=${TIMESTAMP})"
echo "  Artifacts → runs/${DATE_PART}/${TIME_PART}_${VARIANT}/"
echo "  Log directory resolved to: $LOG_DIR/"