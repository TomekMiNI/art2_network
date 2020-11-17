import csv_parser
import MNIST_parser
from artnetwork import ArtNetwork
import numpy as np

n = 784
m = 100
an = ArtNetwork(n, m)

afterExclude = False

size = 100

images_data = MNIST_parser.parse_images("TrainTestData\\train-images.idx3-ubyte")
labels_data = MNIST_parser.parse_labels("TrainTestData\\train-labels.idx1-ubyte")
data = MNIST_parser.convert_image_to_vector(images_data, range(size))
results = np.take(labels_data,np.arange(0,size))

if afterExclude:
    data = MNIST_parser.convert_image_to_vector(images_data, range(size * 1.5))
    results = np.take(labels_data,np.arange(0,size * 1.5))
    
    #exclude class 8 and 9
    dataAfterExclude = [[]] * 100
    iter = 0
    for i in range(size * 1.5):
        if results[i] != 8 and results[i] != 9:
            dataAfterExclude[iter] = data[i]
            iter += 1
            if iter == size:
                data = dataAfterExclude
                break

#data, results = csv_parser.parse_data("SN_projekt2\\klastrowanie\\hexagon.csv", 1)
#data = [[1, 0, 0], [0.99, 0, 0], [0, 1, 0], [0, 0.99, 0], [0, 0, 1], [0, 0, 0.99]]
#results = [0, 0, 1, 1, 2, 2]
an.train(data, 5)
labels_matrix, labels_not_clustered = an.test(data, results)

effectiveness = 0
realCountOfClusters = max(results)
empiricalCountOfClusters = len(labels_matrix)

finalCountOfClusters = min(realCountOfClusters, empiricalCountOfClusters)

checkedClusters = [False] * empiricalCountOfClusters
for rep in range(finalCountOfClusters):
    max_val = -np.inf
    max_i = -1
    max_j = -1
    for i in range(empiricalCountOfClusters):
        if not checkedClusters[i]:
            for j in range(realCountOfClusters):
                if labels_matrix[i][j] > max_val:
                    max_val = labels_matrix[i][j]
                    max_i = i
                    max_j = j
    checkedClusters[max_i] = True
    print(labels_matrix[max_i][max_j], max_i, max_j)
    effectiveness += labels_matrix[max_i][max_j]

print("Effectiveness: ", effectiveness/size)


print("Real classes in network classes")
for i in range(len(labels_matrix)):
    print(labels_matrix[i])
print("Classes not clustered")
print(labels_not_clustered)
