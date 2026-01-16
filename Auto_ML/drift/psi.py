import numpy as np

def psi(expected, actual, buckets=10):
    def scale(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-6)

    expected, actual = scale(expected), scale(actual)
    bins = np.linspace(0, 1, buckets + 1)

    e_hist, _ = np.histogram(expected, bins=bins)
    a_hist, _ = np.histogram(actual, bins=bins)

    e_perc, a_perc = e_hist / len(expected), a_hist / len(actual)

    return np.sum((e_perc - a_perc) * np.log((e_perc + 1e-6)/(a_perc + 1e-6)))


def dataset_psi(train_df, curr_df, threshold=0.2):
    scores = {}
    for col in train_df.select_dtypes(include="number").columns:
        if col == "Survived":
            continue
        scores[col] = psi(train_df[col], curr_df[col])

    return max(scores.values()) > threshold, scores
