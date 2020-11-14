import csv_parser
from artnetwork import ArtNetwork

an = ArtNetwork(2, 20)
data, results = csv_parser.parse_data("SN_projekt2\\klastrowanie\\hexagon.csv", 1)

an.train(data, 1)
labels_matrix, labels_not_clustered = an.test(data, results)
print("Real classes in network classes")
print(labels_matrix)
print("Classes not clustered")
print(labels_not_clustered)
