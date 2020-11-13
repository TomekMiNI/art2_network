import csv_parser
from artnetwork import ArtNetwork

an = ArtNetwork(3, 20)
data, results = csv_parser.parse_data("SN_projekt2\\klastrowanie\\cube.csv", 1)

an.train(data, 1)
print("done")
