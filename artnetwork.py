from typing import List
import numpy as np

class ArtNetwork:
    #weights
    w_weights: [] #weights from input to output
    t_weights: [] #weights from output to input

    #first layer F1
    s: List[int] #read input
    y: List[int] #output
    w: List[int] #pair with x [normalization]
    x: List[float]
    v: List[int] #pair with u [normalization]
    u: List[float]
    p: List[float] #pair with q [normalization]
    q: List[float]

    #parameters
    n: int #size if input
    m: int #size of output
    current_m: int #Initialized by zero and updated to m 
    a: int #fixed weights in the F1 layer au
    b: int #fixed weights in the F1 layer bf(q)
    c: float #fixed weight used in testing for reset 
    d: float #activation of winning F2 unit
    e: float #small parameter used for preventing the division by zero when the vector norm is zero 
    theta: float #parameter  of  noise  suppression
    alfa: float #learning rate; such that small value slows the  learning  and  ensures  that  the  weights reach equilibrium
    ro: float #vigilance parameter 

    def activation_function(self, x):
        result = [0] * len(x)
        for i in range(len(x)):
            if x[i] >= self.theta:
                result[i] = x[i]
        return result

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.current_m = 0
        self.y = [0] * self.m #or put it in the train method

        self.w_weights = [[]] * self.n
        for i in range(self.n):
            self.w_weights[i] = [1/(1+self.n)] * self.m

        self.t_weights = [[]] * self.m #its controlled by self.current_m
        for t_weight in self.t_weights:
            t_weight = [1] * self.n

        self.a = 10
        self.b = 10
        self.c = 0.1
        self.d = 0.9
        self.e = 2.2204e-16
        self.theta = 1/np.sqrt(self.n)
        self.alfa = 0.6
        self.ro = 0.93

    def train(self, inputs, epochs):
        for epoch in range(epochs):
            for s in inputs:
                self.update1(s)
                self.update2(s)
                self.calculateY()
                clustered = False
                for trial in range(self.current_m):
                    J = self.findLargestSignal()
                    if self.verifyForReset(J):
                        self.y[J] = -1
                    else:
                        clustered = True
                        self.update3(s)
                        self.updateWeights(J)
                        self.update4(s, J)
                        #the second condition to stop relates to weight changes - if there is no changes - stop.
                if not clustered:
                    if self.current_m < self.m:
                        #add new class with s as a represent
                        #t_weights[every index] is 1, so lets update with current values
                        self.updateWeights(self.current_m)
                        self.current_m += 1
                    #else: just exclude this input

    
    def update1(self, s):
        self.w = s
        dividerS = self.e + self.norm(s)
        self.x = [el / dividerS for el in s]
        self.v = self.activation_function(self.x) #in other document x is changed later
        self.u = [0] * self.n
        self.p = [0] * self.n
        self.q = [0] * self.n

    def update2(self, s):
        dividerV = self.e + self.norm(self.v)
        self.u = [el / dividerV for el in self.v]
        self.w = s + self.a * self.u
        self.p = self.u
        dividerW = self.e + self.norm(self.w)
        self.x = [el / dividerW for el in self.w]
        dividerP = self.e + self.norm(self.p)
        self.q = [el / dividerP for el in self.p] 
        self.v = self.activation_function(self.x) + self.b * self.activation_function(self.q)

    def update3(self, s):
        self.w = s + self.a * self.u
        dividerW = self.e + self.norm(self.w)
        self.x = [el / dividerW for el in self.w]
        dividerP = self.e + self.norm(self.p)
        self.q = [el / dividerP for el in self.p] 
        self.v = self.activation_function(self.x) + self.b * self.activation_function(self.q)

    def update4(self, s, J):
        dividerV = self.e + self.norm(self.v)
        self.u = [el / dividerV for el in self.v]
        self.w = s + self.a * self.u
        self.p = self.u + [self.d * tJ_i for tJ_i in self.t_weights[J]] # FIX ME :) #the one difference against update2
        dividerW = self.e + self.norm(self.w)
        self.x = [el / dividerW for el in self.w]
        dividerP = self.e + self.norm(self.p)
        self.q = [el / dividerP for el in self.p] 
        self.v = self.activation_function(self.x) + self.b * self.activation_function(self.q)

    def updateWeights(self, J):
        first_multiplier = self.alfa * self.d
        second_multiplier = (1 + self.alfa * self.d * (self.d - 1))
        self.t_weights[J] = [first_multiplier * u_i for u_i in self.u] + [second_multiplier * tJ_i for tJ_i in self.t_weights[J]] # FIX ME (probably) :)
        for i in range(self.n):
            self.w_weights[i][J] = first_multiplier * self.u[i] + second_multiplier * self.w_weights[i][J]

    def calculateY(self):
        for j in range(self.current_m):
            sum = 0
            for i in range(self.n):
                sum += self.w_weights[i][j] * self.p[i]
            self.y[j] = sum

    def findLargestSignal(self):
        largest = -np.inf
        index = -1
        for j in range(self.current_m):
            if self.y[j] != -1 and largest < self.y[j]:
                largest = self.y[j]
                index = j
        return index
                
    def verifyForReset(self, J):
        dividerV = self.e + self.norm(self.v)
        self.u = [el / dividerV for el in self.v]
        self.p = self.u + [self.d * tJ_i for tJ_i in self.t_weights[J]] # FIX ME :)
        dividends = self.u + [self.c * p_i for p_i in self.p] # FIX ME :)
        divider = self.e + self.norm(self.u) + self.c * self.norm(self.p)
        r = [dividend / divider for dividend in dividends]
        return self.norm(r) < self.ro - self.e

    def norm(self, vector):
        result = 0
        for v in vector:
            result += v ** 2
        return result
