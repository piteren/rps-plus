#!/bin/bash
pkill tensorboard
nohup tensorboard --logdir="$PWD/_models" >/dev/null 2>&1 &