#!/bin/sh

./train.py --dataset ../FPHAD --checkpoint ./checkpoints/test --model-def ResNet --inner-steps 1 --algorithm fomaml
