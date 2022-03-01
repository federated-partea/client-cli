import os
import traceback
from partea.api import join_project, run_project
from partea.cli.cli_parser import parse_arguments
from partea.survival_analysis import coxph, univariate

NAME = "Partea CLI Client"


def preprocess_data(input_path: str, method: str, duration_col: str, event_col: str, cond_col: str, sep: str):
    data = None
    cph = None
    if not os.path.exists(input_path):
        print("Path does not exist")
        exit()
    try:
        if method == "univariate":
            data = univariate.preprocess(duration_col, event_col, cond_col, input_path, sep)
        elif method == "cox":
            data = coxph.preprocess(input_path, sep, duration_col, event_col)
            cph = coxph.CoxPHModel(data, duration_col, event_col)
        if data.shape[0] < 5:
            print("You need at least 5 samples to participate in the computation.")
            exit()
        if data is None:
            print("No data was found")
            raise Exception("No data found.")
        return data, cph
    except Exception as e:
        print(f'File could not be processed: {e}')
        traceback.print_exc()
        exit()


class FederatedClient:

    def __init__(self):
        username, token, input_path, password, server_url, duration_col, event_col, category_col, sep = parse_arguments()

        try:
            method, token, client_id, min_time, max_time, step_size, max_iters, smpc = join_project(server_url,
                                                                                                    username, password,
                                                                                                    token)
            print(f'Client ID: {client_id}')
            data, cph = preprocess_data(input_path, method, duration_col, event_col, category_col, sep)

            print('Preprocessing completed. Run project...')
            run_project(method, server_url, token, category_col, data, cph, client_id, min_time, max_time, step_size,
                        smpc)
        except TypeError:
            print("Could not join project with token " + str(token))
            exit()


client = FederatedClient()
