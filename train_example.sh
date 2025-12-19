cd train

while true
do

    CUDA_VISIBLE_DEVICES="0,1,2,3" bash train_muon_ki.sh ../data ../data/shuffleddata/current b14c192h6tfrs_1 b14c192h6tfrs-bng-silu 384 extra -multi-gpus 0,1,2,3 -gnorm-clip-scale 1.0 -lr-scale-auto-type custom -wd-scale 1.0 -export-prob 0.003 -master-port 23456
    
done


cd ..