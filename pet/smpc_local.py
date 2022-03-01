import numpy as np


def make_secure(params: float or int or dict or list, n: int, exp: int) -> list:
    params = to_int(params, exp=exp)
    print(params)
    s_params = create_shards(params, n)

    return s_params


def create_shards(params: float or int or dict or list, n: int) -> list:
    ps = []
    if type(params) == dict:
        for i in range(n):
            ps.append({})
        for key in params.keys():
            pd = create_shards(params[key], n)
            for i in range(len(ps)):
                ps[i][key] = pd[i]
    elif type(params) == list:
        for i in range(n):
            ps.append([])
        for pj in params:
            pd = create_shards(pj, n)
            for i in range(len(ps)):
                ps[i].append(pd[i])
    elif type(params) == float or int:
        rs = int(0)
        for i in range(n - 1):
            r = np.random.randint(1, 100000000000000)
            ps.append(r)
            rs += r

        ps.append(params - rs)
    else:
        raise Exception("This type is not supported. Please only create shards for float, lists or dicts.")

    return ps


def to_int(params: int or float or dict or list, exp) -> int or dict or list:
    p = None
    if type(params) == float or type(params) == int:
        p = int(params * 10 ** exp)
    elif type(params) == list:
        p = []
        for param in params:
            p.append(to_int(param, exp))
    elif type(params) == dict:
        p = {}
        for key in params.keys():
            p[key] = to_int(params[key], exp)
    return p
