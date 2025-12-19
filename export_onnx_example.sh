
cd train
python export_onnx.py -checkpoint ../data/train/b14c192h6tfrs_1/checkpoint.ckpt -export-dir ../data/models_onnx -model-name b14c192h6tfrs_1 -pos-len 19 -batch-size 8 -use-swa -disable-mask