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
    ls: int # learning steps

    def activation_function(self, x):
        result = [0 for _ in range(len(x))]
        for i in range(len(x)):
            if abs(x[i]) >= self.theta:
                result[i] = x[i]
        return result

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.current_m = 0
        self.y = [0] * self.m #or put it in the train method

        self.w_weights = [[]] * self.n
        for i in range(self.n):
            self.w_weights[i] = [1/(1+self.n) for _ in range(self.m)]

        self.t_weights = [[]] * self.m #its controlled by self.current_m
        for i in range(self.m):
            self.t_weights[i] = [0 for _ in range(self.n)]

        self.a = 10
        self.b = 10
        self.c = 0.1
        self.d = 0.9
        self.e = 2.2204e-16
        self.theta = 0.01 #1/np.sqrt(self.n)
        self.alfa = 0.6
        self.ro = 0.91
        self.ls = 5

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
                        for _ in range(self.ls):
                            self.updateWeights(J)
                            self.update4(s, J)
                        break
                        #the second condition to stop relates to weight changes - if there is no changes - stop.
                if not clustered:
                    if self.current_m < self.m:
                        #add new class with s as a represent
                        #t_weights[every index] is 1, so lets update with current values

                        self.updateWeights(self.current_m)
                        self.current_m += 1
                    #else: just exclude this input

    
    def update1(self, s):
        self.w = [si for si in s]
        dividerS = self.e + self.norm(s)
        self.x = [el / dividerS for el in s]
        self.v = self.activation_function(self.x) #in other document x is changed later
        self.u = [0 for _ in range(self.n)]
        self.p = [0 for _ in range(self.n)]
        self.q = [0 for _ in range(self.n)]

    def update2(self, s):
        dividerV = self.e + self.norm(self.v)
        self.u = [el / dividerV for el in self.v]
        for i in range(self.n):
            self.w[i] = s[i] + self.a * self.u[i]
            self.p[i] = self.u[i]

        dividerW = self.e + self.norm(self.w)
        self.x = [el / dividerW for el in self.w]
        dividerP = self.e + self.norm(self.p)
        self.q = [el / dividerP for el in self.p]

        activated_x = self.activation_function(self.x)
        activated_q = self.activation_function(self.q)
        for i in range(self.n):
            self.v[i] = activated_x[i] + self.b * activated_q[i]

    def update3(self, s):
        for i in range(self.n):
            self.w[i] = s[i] + self.a * self.u[i]
        dividerW = self.e + self.norm(self.w)
        self.x = [el / dividerW for el in self.w]
        dividerP = self.e + self.norm(self.p)
        self.q = [el / dividerP for el in self.p]

        activated_x = self.activation_function(self.x)
        activated_q = self.activation_function(self.q)
        for i in range(self.n):
            self.v[i] = activated_x[i] + self.b * activated_q[i]

    def update4(self, s, J):
        dividerV = self.e + self.norm(self.v)
        self.u = [el / dividerV for el in self.v]
        for i in range(self.n):
            self.w[i] = s[i] + self.a * self.u[i]
            self.p[i] = self.u[i] + self.d * self.t_weights[J][i]
        dividerW = self.e + self.norm(self.w)
        self.x = [el / dividerW for el in self.w]
        dividerP = self.e + self.norm(self.p)
        self.q = [el / dividerP for el in self.p]

        activated_x = self.activation_function(self.x)
        activated_q = self.activation_function(self.q)
        for i in range(self.n):
            self.v[i] = activated_x[i] + self.b * activated_q[i]

    def updateWeights(self, J):
        first_multiplier = self.alfa * self.d
        second_multiplier = (1 + self.alfa * self.d * (self.d - 1))

        for i in range(self.n):
            self.t_weights[J][i] = first_multiplier * self.u[i] + second_multiplier * self.t_weights[J][i]
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
        dividends = [0.0] * self.n
        for i in range(self.n):
            self.p[i] = self.u[i] + self.d * self.t_weights[J][i]
            dividends[i] = self.u[i] + self.c * self.p[i]

        divider = self.e + self.norm(self.u) + self.c * self.norm(self.p)
        r = [dividend / divider for dividend in dividends]
        return self.norm(r) < self.ro - self.e

    def norm(self, vector):
        return np.linalg.norm(vector)
        # result = 0
        # for v in vector:
        #    result += v ** 2
        # return np.sqrt(result)

    def test(self, inputs, labels):
        classes = [list() for _ in range(self.current_m)]
        elems_not_in_classes = list()
        idx = 0
        print("Count of clusters: ", self.current_m)
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
                    classes[J].append(idx)
                    break
            if not clustered:
                elems_not_in_classes.append(idx)
            idx += 1

        max_label = max(labels)
        labels_matrix = [[0 for _ in range(max_label + 1)] for _ in range(self.current_m)]
        labels_not_in_classes = [0 for _ in range(max_label + 1)]
        for i in range(self.current_m):
            for j in range(len(classes[i])):
                labels_matrix[i][labels[classes[i][j]]] += 1

        for i in range(len(elems_not_in_classes)):
            labels_not_in_classes[labels[elems_not_in_classes[i]]] += 1

        return labels_matrix, labels_not_in_classes
