"""Dataset loading."""

import os

import numpy as np

UCI_DATASETS = [
    "glass",
    "zoo",
    "iris",
]

def load_codebert_data(points_file, vocab_file):
    """Load CodeBERT activations data."""
    # Load points and vocabulary
    points = np.load(points_file)
    vocab = np.load(vocab_file, allow_pickle=True)
    
    # Calculate similarities using cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(points) # hyPHC needs a similarity matrix
    
        
    # Convert to double precision
    similarities = similarities.astype(np.float64)
    
    return points, vocab, similarities

def load_data(dataset, normalize=True):
    """Load dataset.

    @param dataset: dataset name
    @type dataset: str
    @param normalize: whether to normalize features or not
    @type normalize: boolean
    @return: feature vectors, labels, and pairwise similarities computed with cosine similarity
    @rtype: Tuple[np.array, np.array, np.array]
    """
    if dataset == "codebert":
        points_file = "data/processed_activations/_processed_points.npy"
        vocab_file = "data/processed_activations/_processed_vocab.npy"
        return load_codebert_data(points_file, vocab_file) # cosine similarity matrix done in the load_codebert_data function
    elif dataset in UCI_DATASETS:
        x, y = load_uci_data(dataset)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset))
    if normalize:
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x0 = x[None, :, :]
    x1 = x[:, None, :]
    cos = (x0 * x1).sum(-1)
    similarities = 0.5 * (1 + cos)
    similarities = np.triu(similarities) + np.triu(similarities).T
    similarities[np.diag_indices_from(similarities)] = 1.0
    similarities[similarities > 1.0] = 1.0
    return x, y, similarities


def load_uci_data(dataset):
    """Loads data from UCI repository.

    @param dataset: UCI dataset name
    @return: feature vectors, labels
    @rtype: Tuple[np.array, np.array]
    """
    x = []
    y = []
    ids = {
        "zoo": (1, 17, -1),
        "iris": (0, 4, -1),
        "glass": (1, 10, -1),
    }
    data_path = os.path.join(os.environ["DATAPATH"], dataset, "{}.data".format(dataset))
    classes = {}
    class_counter = 0
    start_idx, end_idx, label_idx = ids[dataset]
    with open(data_path, 'r') as f:
        for line in f:
            split_line = line.split(",")
            
            if len(split_line) >= end_idx - start_idx + 1:
                x.append([float(x) for x in split_line[start_idx:end_idx]])
                label = split_line[label_idx]
                if not label in classes:
                    classes[label] = class_counter
                    class_counter += 1
                y.append(classes[label])
    y = np.array(y, dtype=int)
    x = np.array(x, dtype=float)
    mean = x.mean(0)
    std = x.std(0)
    x = (x - mean) / std
    return x, y
