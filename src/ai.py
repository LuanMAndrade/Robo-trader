import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
import pdb

class Network(nn.Module):
    def __init__ (self,input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions

        # 5 --> 30 --> 3  fc = Full Connection (Dense)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30, nb_actions)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_vallues = self.fc2(x)
        return q_vallues
    
    class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        #samples = zip(*self.memory)
        #return map(lambda x: Variable(torch.cat(x, 0)), samples)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    
    class Dqn():
    def __init__(self, input_size, nb_actions, gamma):
        self.gamma = gamma
        self.input_size = input_size
        self.reward_window = []
        self.model = Network(input_size, nb_actions)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        #Recebe um estado e seleciona uma ação
        with torch.no_grad():
            logits = self.model(Variable(state, volatile = True))*7
            probs = F.softmax(logits)
            action = probs.multinomial(num_samples=1)
        return action.data[0, 0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #É a ação resultado do estado atual aplicado na rede
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #É a ação que seria a "correta" pois é calculada  Q-target = Q(proximo estado)*gamma + recompensa
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        #Função loss aplicada na diferença entre os dois Q
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
        
    def update(self, reward, new_signal):
        #É o estado atual que for recebido pelo cenario, só está sendo feita a conversão do dado 
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #Grava na memoria
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),
                     torch.Tensor([self.last_reward])))
        #Define a próxima ação
        action = self.select_action(new_state)
        #Cria batches a partir dos dados que já estão na memoria e roda a função para aprender
        if len(self.memory.memory) > 100:
            #batch_state, batch_next_state, batch_action = self.memory.sample()
            #deslocamento = []
            #for indice, valor in enumerate(amostra[0:data_size-batch_size+1]):
            #    state.append(valor)
            #    for i in amostra[indice+1:indice+101]:
            #        if i[0] > state[indice][0]+5:
            #            deslocamento.append(1)
            #            break
            #        elif i[0] < state[indice][0]-5:
            #            deslocamento.append(-1)
            #        break
            #batch_reward = 
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)

            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #Atualiza cada dado anterior com seu novo valor
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        #Acrescenta recompensa para ser mostrada no gráfico
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action