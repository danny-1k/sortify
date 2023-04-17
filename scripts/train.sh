cd ../src
nohup python train.py --epochs 100 \
    --lr 1e-5 \
    --batch_size 1 \
    --workers 0 > ../logs/train.txt &