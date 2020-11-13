
from artnetwork import ArtNetwork

an = ArtNetwork(5,3)
inputs = [[1,1,1,0,1],[0,0,1,1,1],[1,0,1,0,1],[1,0,0,1,1]]
an.train(inputs, 1)