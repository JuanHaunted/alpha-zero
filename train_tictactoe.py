import torch
from tictactoe import TicTacToe
from resnet import ResNet
from alphazero import AlphaZero

# Game to train on
ttt = TicTacToe()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet(ttt, 4, 64, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
    'C': 2, 
    'num_searches': 60,
    'num_iterations': 3,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 64,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.30
}

alphaZero = AlphaZero(model, optimizer, ttt, args)
alphaZero.learn()