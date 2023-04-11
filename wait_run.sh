#!/bin/bash
# repeatedly run 'echo hello world'
# tail --pid=$1 -f /dev/null && while true; do sleep 10; python train_dpgan.py; done
tail --pid=$1 -f /dev/null && python train_autoencoder.py

