LXMERT_REPO=https://huggingface.co/datasets/Lo/adapt-pre-trained-VL-models-to-text-data-LXMERT
TMP_DIR=DOWNLOAD_DATA_TMP
LXMERT_DIR_PATH=data/lxmert
WIKIPEDIA_DIR_PATH=data/wikipedia

git lfs install &&\
git clone $LXMERT_REPO ./$TMP_DIR &&\
rm $TMP_DIR/README.md &&\
mv $TMP_DIR/* ./$LXMERT_DIR_PATH &&\
rm -rf $TMP_DIR