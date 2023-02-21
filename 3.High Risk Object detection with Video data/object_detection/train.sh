VERSION=v1
ID=0
LOGFILE=logs/exp_${VERSION}.log
CUDA_VISIBLE_DEVICES=${ID} python3 train.py > "$LOGFILE" 2>&1 &