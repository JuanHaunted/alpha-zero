# AlphaZero - Implementing AlphaZero for Simple Games

## Papers
- AlphaZero: https://arxiv.org/pdf/1712.01815
- MCTS Review: https://arxiv.org/pdf/2103.04931
- ResNet: https://arxiv.org/pdf/1512.03385
- BatchNorm: https://arxiv.org/pdf/1502.03167

## Overview
If you want an overview of the full code, check alpha_zero.ipynb, which contains 
implementation of the complete model with all its components. 

### Training
Use tran_connectfour.py and train_tictactoe.py to train your own weights for the model.
train_tictactoe.py -> Uses GPU/CPU serial version for the AlphaZero algorithm.
train_connectfour.py -> Uses GPU/CPU parallel version of the AlphaZero algorith.

Both models implement most of the AlphaTweaks present in the original AlphaZero paper.

Feel free to alter the training parameters: 

    -'C': 2 -> Exploration parameter for UBC
    -'num_searches': 600 -> Number of searches during MCTS
    -'num_iterations': 8 -> Number of MCTS iterations
    -'num_selfPlay_iterations': 500 -> SelfPlay iterations (For generating syntetic data)
    -'num_epochs': 4 -> Epochs to train (A checkpoint is saved after each epoch)
    -'num_parallel_games': 100 -> Number of games to be simulated during an MCTS iter
    -'batch_size': 128, -> BatchSize used for training the network
    -'temperature': 1.25 -> Increases exploration during training
    -'dirichlet_epsilon': 0.25 -> Dirichlet noise to policies during training
    -'dirichlet_alpha': 0.30 -> Dirichelt noise to policies during training

You can also modify ResNet width and depth. In general use bigger networks for more complex games. 
This has to be manually tuned. 

## Implementing New Games
Both the ConnectFour and TicTacToe classes share the same interface. To implement a new game,
it suffices to create a new Game class that implements this Interface. Return types must be consistent.

In general, the algorithm works for all turn based, discrete games, in which the state can be represented
using three planes. If more planes are needed, modify architecture of Resblocks in the resnet.py to match 
your target implementation. The election of plane representation may require slight modifications to the rest of the code, so real carefully before trying. gGames with more complex rules and game states to be represented. For example chess uses one plane for each piece of each player, and additional planes for representing certain interactions like crowning.

## Eval
Eval scripts are provided to play against the AI through the command line interface. 
Friendly UI interface may be developed in the future. See the eval scripts to see an example
of how to use the models during inference. 


## Special Thanks:
To Proffessor John Levine for his YouTube explaination on Monte Carlo Tree Search. 
To Josh Varty for his explaination on using AlphaZero along with MCTS on YouTube. 
Special thanks to Robert Forester to his detailed implementation guides on the algorithm.
To Aleksa Godric for his in detail walkthrough of the original paper on YouTube. 
To the members of DeepMind for making this algo public.

