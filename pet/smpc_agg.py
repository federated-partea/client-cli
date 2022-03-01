def apply_operation(params, operation: str = 'add'):
    """

    :param params:
    :param operation:
    :return:
    """
    if operation not in ['add']:
        return None
    s = []
    for i in range(len(params[0])):
        if isinstance(params[0][i], float) or isinstance(params[0][i], int):
            p = 0.0
            for param in params:
                p += param[i]
            s.append(p)
        elif isinstance(params[0][i], list):
            ps = []
            for param in params:
                ps.append(param[i])
            p = apply_operation(ps)
            s.append(p)
        elif isinstance(params[0][i], dict):
            p = {}
            for key in params[0][i].keys():
                ps = []
                for param in params:
                    try:
                        ps.append([param[i][key]])
                    except KeyError:
                        ps.append([0.0])
                p[key] = apply_operation(ps)
                if len(p[key]) == 1:
                    p[key] = p[key][0]
            s.append(p)

    return s


def sum_dicts(dicts: [dict]):
    return {k: sum(d[k] for d in dicts) for k in dicts[0]}


def aggregate_smpc(params, operation='add'):
    """
    Aggregates parameters into a new parameter struct based on the specified operation
    :param params:
    :param operation:
    :return:
    """
    if len(params) > 1:
        agg = apply_operation(params=params, operation=operation)
    else:
        return params
    return agg


def distribute_smpc(data: dict):
    distribute_dict = {}
    i = 0
    for key in data.keys():
        client_dist = []
        for key2 in data.keys():
            client_dist.append([data[key2]["data"][i]])
        i += 1
        distribute_dict[key] = client_dist

    return distribute_dict
