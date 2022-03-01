import pickle


def serialize(p):
    return pickle.dumps(p)


def deserialize(p):
    return pickle.loads(p)
