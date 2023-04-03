MODELS_DIR_PATH=adaptations/data/runs/finetune
WEIGHTS_REPO=https://huggingface.co/Lo/adapt-pre-trained-VL-models-to-text-finetuned-weights

#git lfs install &&\
#rm -rf TMP &&\
#git clone $WEIGHTS_REPO ./TMP &&\
#rm TMP/README.md &&\
cp -r TMP/* ./$MODELS_DIR_PATH &&\
rm -rf TMP
