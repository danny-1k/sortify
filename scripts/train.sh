cd ../src
nohup python train.py --epochs 100 \
    --lr 2e-3 \
    --batch_size 30 \
    --workers 0 > ../logs/train.txt &