import csv
from typing import List


def parse_data(file_name: str, headers_row_count: int) -> (List[List[float]], List[List[float]]):
    with open(file_name) as f:
        reader = csv.reader(f)
        loaded_data = list(reader)

    data_count = len(loaded_data[0]) - 1
    data = list()
    results = list()
    for i in range(headers_row_count, len(loaded_data)):
        data.append(loaded_data[i][:data_count])
        data[i - 1] = [float(i) for i in data[i - 1]]
        results.append(loaded_data[i][data_count:])
        results[i - 1] = [int(i) for i in results[i - 1]]

    return data, results
