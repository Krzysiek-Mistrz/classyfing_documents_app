import pickle

def save_list(obj, filename):
    """Persist Python object (e.g. list) to disk."""
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def load_list(filename):
    """Load Python object from disk."""
    with open(filename, "rb") as f:
        return pickle.load(f)