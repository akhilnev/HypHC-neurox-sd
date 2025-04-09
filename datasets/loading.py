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
    
    # Calculate similarities using batched cosine similarity to save memory
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Try to use a smaller subset for initial testing
    if points.shape[0] > 20000:
        print(f"WARNING: Large dataset with {points.shape[0]} points detected.")
        print("Computing similarities in batches to save memory...")
        
        # Compute in batches
        BATCH_SIZE = 1000
        n_samples = points.shape[0]
        similarities = np.zeros((n_samples, n_samples), dtype=np.float32)  # Use float32 initially
        
        for i in range(0, n_samples, BATCH_SIZE):
            end_i = min(i + BATCH_SIZE, n_samples)
            batch = points[i:end_i]
            print(f"Computing similarities for batch {i}-{end_i}/{n_samples}")
            batch_similarities = cosine_similarity(batch, points)
            similarities[i:end_i] = batch_similarities
    else:
        similarities = cosine_similarity(points)
    
    # Convert to double precision only at the end
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
