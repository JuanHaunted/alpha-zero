from connectfour import ConnectFour
import torch
from resnet import ResNet
from alphazeroparallel import AlphaZeroParallel

game = ConnectFour()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# model = ResNet(game, 4, 64, device=device)
model = ResNet(game, 9, 128, device=device) # Use for connect fout

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


args = {
    'C': 2, 
    'num_searches': 600,
    'num_iterations': 8,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'num_parallel_games': 100,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.30
}

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()