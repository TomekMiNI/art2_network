import csv_parser
import MNIST_parser
from artnetwork import ArtNetwork
import MNIST_parser
import numpy as np

<<<<<<< HEAD
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
=======
an = ArtNetwork(784, 20)

images = MNIST_parser.parse_images("train-images.idx3-ubyte")
images_vectors = MNIST_parser.convert_image_to_vector(images, range(len(images)))

test_images = MNIST_parser.parse_images("t10k-images.idx3-ubyte")
test_labels = MNIST_parser.parse_labels("t10k-labels.idx1-ubyte")
test_images_vectors = MNIST_parser.convert_image_to_vector(test_images, range(len(test_images)))
an.train(images_vectors, 1)
labels_matrix, labels_not_clustered = an.test(test_images_vectors, test_labels)
>>>>>>> 6d8b03cf0dafe3bed90bd70a6d94290f00e683e9
print("Real classes in network classes")
for i in range(len(labels_matrix)):
    print(labels_matrix[i])
print("Classes not clustered")
print(labels_not_clustered)
