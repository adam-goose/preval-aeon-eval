#!/bin/bash
# Focused diagnostic experiment for Iridis. Edit the configuration block only.
set -euo pipefail

# ---------------- configuration ----------------
username="${USER}"
repo_dir="/iridisfs/home/${username}/preval-aeon-eval"
data_dir="/iridisfs/scratch/${username}/Data"
cache_dir="/iridisfs/scratch/${username}/preval_diagnostic_feature_cache"
output_dir="${repo_dir}/analysis/ucr112_resample0/failure_investigation/diagnostic_experiment/results"
env_name="tsml-eval"
queue="batch"
memory="32000M"
walltime="24:00:00"
cpus=4
# ------------------------------------------------

script_dir="${repo_dir}/analysis/ucr112_resample0/failure_investigation/diagnostic_experiment"
cases_file="${script_dir}/cases.csv"
log_dir="${script_dir}/logs"
mkdir -p "${log_dir}" "${cache_dir}" "${output_dir}"
n_cases=$(( $(wc -l < "${cases_file}") - 1 ))

submission="${script_dir}/diagnostic_array.sub"
cat > "${submission}" <<EOF
#!/bin/bash
#SBATCH -p ${queue}
#SBATCH -t ${walltime}
#SBATCH --mem=${memory}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --array=1-${n_cases}
#SBATCH --job-name=PreValDiag
#SBATCH -o ${log_dir}/%A-%a.out
#SBATCH -e ${log_dir}/%A-%a.err

set -euo pipefail
. /etc/profile
module load conda/python3
source activate ${env_name}

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" "${cases_file}")
IFS=',' read -r dataset transform role original_delta <<< "\${line}"
complete="${output_dir}/\${transform}/\${dataset}/COMPLETE"
if [[ -f "\${complete}" ]]; then
    echo "Already complete: \${transform}/\${dataset}"
    exit 0
fi

python -u "${script_dir}/run_diagnostic.py" "\${dataset}" "\${transform}" \
    --data-dir "${data_dir}" \
    --cache-dir "${cache_dir}" \
    --output-dir "${output_dir}" \
    --design full \
    --fixed-lambda 1.0 \
    --n-jobs "${cpus}" \
    --save-predictions
EOF

echo "Prepared ${submission} with ${n_cases} array tasks."
echo "Feature cache: ${cache_dir}"
echo "Submit with: sbatch ${submission}"
sbatch "${submission}"
