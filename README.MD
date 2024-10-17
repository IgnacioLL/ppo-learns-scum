# PPO Agent plays Scum

## Introduction

Scum, also known as President or Speed Capitalism, is a card game for 4-6 players that is similar to crazy eights. This game is played with a standard 52 card deck. Cards are played in equal number (i.e. only singles on singles, pairs on pairs, triplets on triplets) and cannot be of a lower rank (i.e. no playing a 6 on an 8). The first player to get rid of all their cards wins the hand. Play continues for as many hands as people feel like playing. [1](https://www.realcty.org/wiki/Scum)

## Objective

The aim of this repository is to create an Actor Critic agent which solves the optimal way of playing scum by using deep Reinforcement Learning.

## Setup

To download all the necesary dependencies in an AWS EC2 instance use `bash setup/installation.sh` and `pip install -r setup/requirements.txt`.

## Training

To train the algorithm and tune the hyperparameters use `python src/hypertuner.py` this will create multiple trainings with some hyperpamaters tweaks to check which is better. (TODO)