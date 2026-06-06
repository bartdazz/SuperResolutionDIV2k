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
tree = ast.parse(src)

# Collect simple constants that VARIANT may reference (e.g. SCALE_FACTOR = 2)
consts = {}
for n in ast.walk(tree):
    if isinstance(n, ast.Assign):
        for t in n.targets:
            if isinstance(t, ast.Name):
                try:
                    consts[t.id] = ast.literal_eval(n.value)
                except Exception:
                    pass

for n in ast.walk(tree):
    if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == 'VARIANT' for t in n.targets
    ):
        try:
            val = eval(
                compile(ast.Expression(n.value), '<string>', 'eval'),
                {'__builtins__': None, 'str': str},
                consts,
            )
            print(val); sys.exit()
        except Exception:
            pass
sys.exit(1)
")

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
DATE_PART=${TIMESTAMP%%_*}
TIME_PART=${TIMESTAMP#*_}
LOG_DIR="../slurm_logs/${DATE_PART}/${TIME_PART}_${VARIANT}"
mkdir -p "$LOG_DIR"
# Resolve to absolute path after ensuring directory exists
LOG_DIR="$(cd "$LOG_DIR" && pwd)"


sbatch \
    --output="${LOG_DIR}/slurm_%j.out" \
    --error="${LOG_DIR}/slurm_%j.err" \
    <<EOF
#!/bin/bash -l
#SBATCH --job-name=${VARIANT}
#SBATCH --partition=testing
#SBATCH --account=thesis
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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