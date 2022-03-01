import traceback
from time import sleep

import pandas as pd
import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import json
from requests.exceptions import ConnectionError
import numpy as np

from partea.survival_analysis import univariate
from partea.survival_analysis.coxph import CoxPHModel
from pet.encryption import Encryption, encrypt_outgoing
from pet.smpc_agg import aggregate_smpc
from pet.smpc_local import make_secure
from serialize import deserialize, serialize

retry_strategy = Retry(
    total=20,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"],
    backoff_factor=1
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

headers = {
    'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/86.0.4240.111 Safari/537.36'}


def join_project(server_url: str, username: str, password: str, token: str):
    request_body = {'username': username, 'password': password, 'token': token}
    try:
        response = http.post(f'{server_url}/project/', json=request_body)
        if response.status_code != 200:
            error_message = response.content.decode('utf-8')
            print("Join Error:", error_message)
            exit()

        response_data = json.loads(response.content)
        method = response_data['project']['method']
        min_time = response_data['project']['from_time']
        max_time = response_data['project']['to_time']
        step_size = response_data['project']['step_size']
        smpc = response_data['project']['smpc']
        max_iters = response_data['project']['max_iters']

        client_id = response_data['client_id']
        token = response_data['token']

        print(f'Perform {method} computation')

        return method, token, client_id, min_time, max_time, step_size, max_iters, smpc

    except ConnectionError as e:
        print("Connection Error", "Connection to server could not be established.", e)
        traceback.print_exc()
        exit()


def get_global_data(token: str, client_step: int, server_url: str, client_id: int = 0):
    while True:
        response = http.get(url=f'{server_url}/task/?token={token}&mode=state', headers=headers)
        data_from_server = json.loads(response.content)

        server_state = data_from_server['state']
        server_step = int(data_from_server['step'])
        project_state = data_from_server['internal_state']

        if (server_state == 'waiting') and server_step > client_step:
            if project_state == "init":
                return {}, server_step, "init"
            response = http.get(url=f'{server_url}/task/?token={token}&mode=data&client={str(client_id)}',
                                headers=headers)
            if response.status_code == 200:
                data = deserialize(response.content)
                client_step = server_step
                sleep(1)
                if data is not None:
                    if isinstance(data, dict):
                        if "error" in data.keys():
                            raise RuntimeError(f'Error during computation: {data["error"]}')
                    return data, client_step, project_state
            elif response.status_code == 500:
                print(response.content)
                print(response.status_code)

        else:
            sleep(1)
            continue


def send_local_data(data, token: str, server_url: str):
    serialized_local_data = serialize(data)
    response = http.post(url=f'{server_url}/task/?token={token}', data=serialized_local_data, headers=headers)
    if response.status_code == 200:
        sleep(1)
        return
    else:
        print("Could not send data to server. Try again.", response.status_code, response.content)


def wait_until_project_started(token: str, server_url: str):
    print("Wait for project start...")
    while True:
        response = http.get(f'{server_url}/task/?token={token}&mode=state', headers=headers)
        decoded_resp = json.loads(response.content)
        if decoded_resp['state'] == 'waiting':
            sleep(1)
            return
        print("still waiting...")
        sleep(3)


def run_project(method: str, server_url: str, token: str, category_col: str, data: pd.DataFrame, cph: CoxPHModel,
                client_id: int, min_time: float, max_time: float, step_size: float, smpc: bool):
    try:
        wait_until_project_started(token, server_url)
        if method == "univariate":
            if not smpc:
                univariate_analysis(data, category_col, token, server_url)
            else:
                smpc_univariate_analysis(data, category_col, token, server_url, min_time, max_time, step_size,
                                         client_id)
        elif method == "cox":
            if not smpc:
                regression_analysis(data, token, server_url, cph)
            else:
                smpc_regression_analysis(cph, token, server_url, min_time, max_time, step_size, client_id)
    except Exception as e:
        print("Computation Error.", "The computation failed.")
        traceback.print_exc()
        try:
            send_local_data({"error": e}, token, server_url)
        except urllib3.exceptions.MaxRetryError:
            print(
                'The computation failed. The partea server was not reachable. Check the webapp for more informations.')
            exit()

        print('The computation failed. The coordinator needs to create a new project.')
        exit()


def univariate_analysis(data: pd.DataFrame, category_col: str, token: str, server_url: str):
    print("Start univariate computation...")
    state = None
    client_step = 0
    while state != "finished":
        global_data, client_step, state = get_global_data(token, client_step, server_url)
        print(state)

        local_results = univariate.compute(category_col, data)
        send_local_data({"local_results": local_results, "sample_number": data.shape[0]}, token, server_url)
        state = "finished"
        print("Computation Finished.")
        exit()


def smpc_univariate_analysis(local_data: pd.DataFrame, category_col: str, token: str, server_url: str, min_time: float,
                             max_time: float, step_size: float, client_id: int):
    print("Start univariate computation with secure aggregation...")
    state = None
    client_step = 0
    enc = Encryption()
    while state != "finished":
        data, client_step, state = get_global_data(token, client_step, server_url, client_id)
        print(state)
        if state == "init":
            send_local_data(enc.public_key, token, server_url)
        elif state == "smpc_agg":
            decrypted_data = enc.decrypt_incoming(data)
            local_smpc_aggregate = aggregate_smpc(decrypted_data)
            send_local_data(local_smpc_aggregate, token, server_url)
        elif state == "local_calc":
            enc.public_keys = data
            participants = list(enc.public_keys.keys())
            local_results = univariate.compute(category_col, local_data, min_time, max_time, step_size)
            smpc_data = make_secure(params={"local_results": local_results, "sample_number": local_data.shape[0]},
                                    n=len(participants), exp=0)
            encrypted_data = encrypt_outgoing(data=smpc_data, public_keys=enc.public_keys)
            send_local_data(encrypted_data, token, server_url)
        else:
            state = "finished"
            print("Computation Finished.")
            exit()


def regression_analysis(data: pd.DataFrame, token: str, server_url: str, cph: CoxPHModel):
    print("Start regression computation...")
    state = None
    client_step = 0

    while state != "finished":

        global_data, client_step, state = get_global_data(token, client_step, server_url)
        print(state)
        if state == "init":
            mean = cph.get_mean()
            send_local_data({"mean": mean, "n_samples": cph.n_samples}, token, server_url)
        elif state == "norm_std":
            cph.set_mean(global_data["norm_mean"])
            std = cph.get_std(cph.get_mean())
            send_local_data({"std": std, "n_samples": cph.n_samples}, token, server_url)
        elif state == "local_init":
            cph.set_std(global_data["norm_std"])
            cph.normalize_local_data()
            distinct_times, zlr, numb_d_set, n_samples = cph._local_initialization()
            send_local_data(
                {"distinct_times": distinct_times, "zlr": zlr, "numb_d_set": numb_d_set, "n_samples": n_samples}, token,
                server_url)
        elif state == "iteration_update":
            if "params" not in global_data.keys():
                print(f'UPDATE BETA - iteration {global_data["iteration"]}')

                i1, i2, i3 = cph._update_aggregated_statistics_(global_data["beta"])

                if np.nan in list(i1.values()) or np.nan in list(i1.values()) or np.nan in list(i1.values()):
                    raise RuntimeError("Convergence error. Could not compute the beta updates.")
                if np.inf in list(i1.values()) or np.inf in list(i1.values()) or np.inf in list(i1.values()):
                    raise RuntimeError("Convergence error. Could not compute the beta updates.")

                send_local_data({"is": [i1, i2, i3]}, token, server_url)
        elif state == "finished":
            cph.params_ = global_data["params"]
            c_index = cph.local_concordance_calculation()
            send_local_data({"c-index": c_index, "sample_number": data.shape[0]}, token, server_url)
            state = "finished"
            print("Computation finished")
            sleep(2)
            exit()


def smpc_regression_analysis(cph: CoxPHModel, token: str, server_url: str, min_time: float,
                             max_time: float, step_size: float, client_id: int):
    exp = 10
    print("Start regression computation with secure aggregation...")
    state: str = None
    client_step = 0
    enc = Encryption()
    participants = None
    while state != "finished":
        data, client_step, state = get_global_data(token, client_step, server_url, client_id)
        print(state)
        if state == "init":
            cph.smpc = True
            send_local_data(enc.public_key, token, server_url)
        elif state.startswith("smpc_agg"):
            decrypted_data = enc.decrypt_incoming(data)
            local_smpc_aggregate = aggregate_smpc(decrypted_data)
            send_local_data(local_smpc_aggregate, token, server_url)
        elif state == "norm_mean":
            enc.public_keys = data
            participants = list(enc.public_keys.keys())
            n_samples = cph.X.shape[0]
            mean = cph.get_mean() * n_samples
            smpc_data = make_secure(params={"mean": mean.to_dict(), "n_samples": n_samples/10**exp}, n=len(participants),
                                    exp=exp)
            encrypted_data = encrypt_outgoing(data=smpc_data, public_keys=enc.public_keys)
            send_local_data(encrypted_data, token, server_url)
        elif state == "norm_std":
            cph.set_mean(data)
            std = cph.get_std(cph.get_mean())
            smpc_data = make_secure(params={"std": std.to_dict()}, n=len(participants), exp=exp)
            encrypted_data = encrypt_outgoing(data=smpc_data, public_keys=enc.public_keys)
            send_local_data(encrypted_data, token, server_url)
        elif state == "local_init":
            cph.set_std(data)
            cph.normalize_local_data()
            timeline = np.arange(min_time, max_time, step_size).tolist()
            timeline.reverse()
            cph.timeline = timeline
            _, zlr, numb_d_set, n_samples = cph._local_initialization()
            smpc_data = make_secure(params={"zlr": zlr.to_dict(), "numb_d_set": numb_d_set, "n_samples": n_samples},
                                    n=len(participants), exp=exp)
            encrypted_data = encrypt_outgoing(data=smpc_data, public_keys=enc.public_keys)
            send_local_data(encrypted_data, token, server_url)
        elif state == "iteration_update":
            print(f'update beta - iteration {data[1]}')
            i1, i2, i3 = cph._update_aggregated_statistics_(beta=data[0])

            smpc_data = make_secure(params={"i1": i1, "i2": i2, "i3": i3}, n=len(participants), exp=exp)
            encrypted_data = encrypt_outgoing(data=smpc_data, public_keys=enc.public_keys)
            send_local_data(encrypted_data, token, server_url)
        elif state == "c_index":
            cph.params_ = data
            n_samples = cph.X.shape[0]
            c_idx = cph.local_concordance_calculation() * 10 * n_samples
            smpc_data = make_secure(params={"c-index": c_idx}, n=len(participants), exp=0)
            encrypted_data = encrypt_outgoing(data=smpc_data, public_keys=enc.public_keys)
            send_local_data(encrypted_data, token, server_url)
        elif state == "finished":
            print("Computation Finished.")
            exit()
        else:
            print("Unknown state: " + state)
