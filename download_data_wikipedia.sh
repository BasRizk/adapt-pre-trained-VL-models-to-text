WIKIPEDIA_DIR_PATH=data/wikipedia

cd $WIKIPEDIA_DIR_PATH &&\
python generate_subset.py --dataset-name "wikipedia" --revision "20200501.en" --save-dir "." --cache-dir "cache" --nbr-train-samples 4400000 --nbr-val-samples 100000 &&\
cd ../..

