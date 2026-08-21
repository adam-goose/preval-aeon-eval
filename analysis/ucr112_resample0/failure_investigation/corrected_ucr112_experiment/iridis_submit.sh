#!/bin/bash
set -euo pipefail

# Corrected UCR112 resample-0 run: 6 classifiers x 112 datasets = 672 tasks.
# Edit only this configuration block for the Iridis account/environment.
username="${IRIDIS_USERNAME:-${USER:-user}}"
repo_dir="${TSML_EVAL_REPO:-/iridisfs/home/${username}/tsml-eval}"
data_dir="${UCR_DATA_DIR:-/iridisfs/home/${username}/Data}"
results_dir="${RESULTS_DIR:-/iridisfs/home/${username}/CorrectedPreValUCR112/results}"
logs_dir="${LOGS_DIR:-/iridisfs/home/${username}/CorrectedPreValUCR112/logs}"
env_name="${CONDA_ENV:-tsml-eval}"
queue="${SLURM_QUEUE:-batch}"
cpus="${CPUS_PER_TASK:-8}"
memory="${MEMORY:-32G}"
walltime="${WALLTIME:-60:00:00}"
max_concurrent="${MAX_CONCURRENT:-100}"

script_dir="$(cd "${BASH_SOURCE[0]%/*}" && pwd)"
classifiers_file="${script_dir}/classifiers.txt"
datasets_file="${script_dir}/ucr112_datasets.txt"
runner="${repo_dir}/tsml_eval/experiments/threaded_classification_experiments.py"

mapfile -t classifiers < "${classifiers_file}"
mapfile -t datasets < "${datasets_file}"
n_classifiers="${#classifiers[@]}"
n_datasets="${#datasets[@]}"
n_tasks="$((n_classifiers * n_datasets))"

if [[ "${n_classifiers}" -ne 6 || "${n_datasets}" -ne 112 ]]; then
    echo "Expected 6 classifiers and 112 datasets; found ${n_classifiers} and ${n_datasets}." >&2
    exit 1
fi

mkdir -p "${logs_dir}"
submission_file="${script_dir}/corrected_ucr112_array.sub"

cat > "${submission_file}" <<EOF
#!/bin/bash
#SBATCH --job-name=PreValCorrected112
#SBATCH --partition=${queue}
#SBATCH --time=${walltime}
#SBATCH --mem=${memory}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-$((n_tasks - 1))%${max_concurrent}
#SBATCH --output=${logs_dir}/%A-%a.out
#SBATCH --error=${logs_dir}/%A-%a.err

set -euo pipefail
. /etc/profile
module load conda/python3
source activate ${env_name}

mapfile -t classifiers < "${classifiers_file}"
mapfile -t datasets < "${datasets_file}"
classifier_index=\$((SLURM_ARRAY_TASK_ID / ${n_datasets}))
dataset_index=\$((SLURM_ARRAY_TASK_ID % ${n_datasets}))
classifier="\${classifiers[\${classifier_index}]}"
dataset="\${datasets[\${dataset_index}]}"

python -u "${runner}" "${data_dir}" "${results_dir}" \
    "\${classifier}" "\${dataset}" 0 -nj ${cpus}
EOF

echo "Prepared ${n_tasks} tasks in ${submission_file}."
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: not submitting."
else
    sbatch "${submission_file}"
fi
