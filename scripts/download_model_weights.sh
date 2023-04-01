MODELS_DIR_PATH=models/data/model-weights
WEIGHTS_REPO=https://huggingface.co/Lo/measure-visual-commonsense-knowledge-model-weights

git lfs install &&\
rm -rf TMP &&\
git clone $WEIGHTS_REPO ./TMP &&\
rm TMP/README.md &&\
mv TMP/* ./$MODELS_DIR_PATH &&\
rm -rf TMP
