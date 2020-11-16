import csv_parser
from artnetwork import ArtNetwork
import MNIST_parser
import numpy as np

an = ArtNetwork(784, 10)

images_data = MNIST_parser.parse_images("TrainTestData\\train-images.idx3-ubyte")
labels_data = MNIST_parser.parse_labels("TrainTestData\\train-labels.idx1-ubyte")
data = MNIST_parser.convert_image_to_vector(images_data, range(100))
results = np.take(labels_data,np.arange(0,100))

#data, results = csv_parser.parse_data("SN_projekt2\\klastrowanie\\hexagon.csv", 1)
#data = [[1, 0, 0], [0.99, 0, 0], [0, 1, 0], [0, 0.99, 0], [0, 0, 1], [0, 0, 0.99]]
#results = [0, 0, 1, 1, 2, 2]
an.train(data, 15)
labels_matrix, labels_not_clustered = an.test(data, results)
print("Real classes in network classes")
for i in range(len(labels_matrix)):
    print(labels_matrix[i])
print("Classes not clustered")
print(labels_not_clustered)
