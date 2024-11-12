from tictactoe import TicTacToe
from connectfour import ConnectFour
import torch
import matplotlib.pyplot as plt
from resnet import ResNet

ttt = TicTacToe()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

state = ttt.get_initial_state()
state = ttt.get_next_state(state, 0, 1)
state = ttt.get_next_state(state, 8, -1)
state = ttt.get_next_state(state, 6, 1)
state = ttt.get_next_state(state, 3, -1)
state = ttt.get_next_state(state, 2, 1)


encoded_state = ttt.get_encoded_state(state)

print(state)
 
tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(ttt, 4, 64, device=device)
model.load_state_dict(torch.load('../checkpoints_TicTacToe/model_2_TicTacToe.pt'))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value, policy)

plt.bar(range(ttt.action_size), policy)
# add title, y_axis label, and x_axis label to the plot
plt.title('Policy (S)')
plt.ylabel('Probability')
plt.xlabel('Action')
# add annotation with the value in the top right part of the plot
plt.annotate(f'V(S) = {value:.2f}', xy=(0.75, 0.75), xycoords='axes fraction')
plt.show()