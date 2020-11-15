import csv_parser
from artnetwork import ArtNetwork

an = ArtNetwork(2, 20)
data, results = csv_parser.parse_data("SN_projekt2\\klastrowanie\\hexagon.csv", 1)
#data = [[1, 0, 0], [0.99, 0, 0], [0, 1, 0], [0, 0.99, 0], [0, 0, 1], [0, 0, 0.99]]
#results = [0, 0, 1, 1, 2, 2]
an.train(data, 10)
labels_matrix, labels_not_clustered = an.test(data, results)
print("Real classes in network classes")
for i in range(len(labels_matrix)):
    print(labels_matrix[i])
print("Classes not clustered")
print(labels_not_clustered)
