ENV_NAME=vl

module load conda/4.12.0
module load gcc/11.3.0
module load git/2.36.1
module load git-lfs/3.2.0

source activate $ENV_NAME

echo "Done init Env with $ENV_NAME"