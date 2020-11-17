import numpy as np
import struct


def parse_images(filename):
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))

    return data


def parse_labels(filename):
    with open(filename, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape(size)

    return data


def get_first_elems_count(labels, elemes_count):
    selected_indices = list()
    for i in range(10):
        selected_indices = selected_indices + get_first_labels(labels, i, elemes_count)

    return selected_indices


def get_first_labels(labels, label, elems_count):
    indices = [None] * elems_count
    labels_len = len(labels)
    selected_items = 0
    cur_idx = 0
    while cur_idx < labels_len and selected_items < elems_count:
        if labels[cur_idx] == label:
            indices[selected_items] = cur_idx
            selected_items = selected_items + 1

        cur_idx = cur_idx + 1

    return indices


def convert_image_to_vector(images, indices):
    vectors = list()
    for i in indices:
        image = images[i]
        image = image.flatten()
        image = image / 255
        vectors.append(image)

    return vectors


def convert_label_to_vector(labels, indices):
    labels_vectors = list()
    for i in indices:
        label = [0] * 10
        label[labels[i]] = 1
        labels_vectors.append(label)

    return labels_vectors
