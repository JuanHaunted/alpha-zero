from alphaMCTS import AMCTS
from tictactoe import TicTacToe
import torch
from resnet import ResNet
import numpy as np
import time

game = TicTacToe()
player = 1

times = [(50, 0.348), 
         (100, 0.568), 
         (150, 0.834), 
         (200, 1.108), 
         (250, 1.618), 
         (300, 1.218), 
         (350, 1.464), 
         (400, 1.190), 
         (450, 1.689), 
         (500, 2.604), 
         (550, 1.571), 
         (600, 1.595) ]

args = {
    'C': 2,
    'num_searches': 300,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 4, 64, device)
model.load_state_dict(torch.load("checkpoints_TicTacToe/model_2_TicTacToe.pt", map_location=device))
model.eval()

mcts = AMCTS(game, args, model)

state = game.get_initial_state()

start = 0
end = 0

while True:
    print(state)
    
    if player == 1:
        valid_moves = game.get_valid_moves(state)
        print("valid_moves", [i for i in range(game.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("action not valid")
            continue
            
    else:
        neutral_state = game.change_perspective(state, player)
        start = time.time()
        mcts_probs = mcts.search(neutral_state)
        end = time.time()
        action = np.argmax(mcts_probs)
        
    print(end - start)
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
        else:
            print("draw")
        break
        
    player = game.get_opponent(player)