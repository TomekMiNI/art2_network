import csv_parser
import MNIST_parser
from artnetwork import ArtNetwork

an = ArtNetwork(784, 20)

images = MNIST_parser.parse_images("train-images.idx3-ubyte")
images_vectors = MNIST_parser.convert_image_to_vector(images, range(len(images)))

test_images = MNIST_parser.parse_images("t10k-images.idx3-ubyte")
test_labels = MNIST_parser.parse_labels("t10k-labels.idx1-ubyte")
test_images_vectors = MNIST_parser.convert_image_to_vector(test_images, range(len(test_images)))
an.train(images_vectors, 1)
labels_matrix, labels_not_clustered = an.test(test_images_vectors, test_labels)
print("Real classes in network classes")
for i in range(len(labels_matrix)):
    print(labels_matrix[i])
print("Classes not clustered")
print(labels_not_clustered)
