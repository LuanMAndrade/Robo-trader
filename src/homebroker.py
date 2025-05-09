import yfinance as yf
import pandas as pd

# Baixar dados do Yahoo Finance
ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01', interval='1d')

# Drop de NaN
data = data.dropna()

# Espaço reservado para tratar os dados



# Aqui eu vou colocar tudo que dê pra mudar para testar
gamma = 0.9 # taxa de desconto
n_estados = 10 # número de informações que a rede neural vai receber.
#Pegar todos os dados tratados e colocar em um dataframe, será o dataframe do estado

brain = Dqn(n_estados,3,gamma) # Instanciamos a rede neural, precisa ainda colocar os parâmetros
action = [1,0,-1] # action = 0 => sem rotação, action = 1 => rotaciona 20 graus, action = 2 => rotaciona -20 graus
scores = [] # inicialização do valor médio das recompensas (sliding window) com relação ao tempo