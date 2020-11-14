import csv_parser
from artnetwork import ArtNetwork

an = ArtNetwork(2, 20)
data, results = csv_parser.parse_data("SN_projekt2\\klastrowanie\\hexagon.csv", 1)

an.train(data, 1000)
print("done")
