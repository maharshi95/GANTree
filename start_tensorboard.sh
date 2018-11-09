#!/usr/bin/env bash
cd $1
echo -e "\e[96m\e[1m\e[4mTensorboard starting @ http://10.24.32.52:$2\e[0m"
tensorboard --logdir=logs --port $2