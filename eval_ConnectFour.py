from alphaMCTS import AMCTS
from connectfour import ConnectFour
import torch
from resnet import ResNet
import numpy as np
import time

game = ConnectFour()
player = 1

turn = 1
times = []

args = {
    'C': 2,
    'num_searches': 100,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device)
model.load_state_dict(torch.load("checkpoints_ConnectFour/model_7_ConnectFour.pt", map_location=device))
model.eval()

mcts = AMCTS(game, args, model)

state = game.get_initial_state()


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
        times.append((end - start ,turn))
        turn += 1
        action = np.argmax(mcts_probs)
        
    state = game.get_next_state(state, action, player)
    
    value, is_terminal = game.get_value_and_terminated(state, action)
    
    if is_terminal:
        print(state)
        if value == 1:
            print(player, "won")
            print(times)
        else:
            print("draw")
            print(times)
        break
        
    player = game.get_opponent(player)