import socket
import pickle
import time
import threading
import random
import numpy as np
import pandas as pd
from pyhocon import ConfigFactory
from queue import Queue
from threading import Thread
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score


class FedAVGServer(threading.Thread):
    def __init__(self, address, port, buffer_size, training_file, timeout, rounds_limit, n_expected_clients,
                 c_fraction, convergence_score):

        threading.Thread.__init__(self)
        self.socket = None
        self.address = address
        self.port = port
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.rounds_limit = rounds_limit
        self.n_expected_clients = n_expected_clients
        self.c_fraction = c_fraction
        self.convergence_score = convergence_score
        self.k = 3
        self.training_file = training_file
        self.training_data = None

        # Specification of column names present on the training dataset (name for each feature from each conversation
        # identified by layer 2)
        self.__all_columns_names = ["Uplink_IP", "Downlink_IP", "Uplink_port", "Downlink_port",
                                    "Duration", "Packets_up", "Packets_down", "Bytes_up", "Bytes_down",
                                    "Max_ps_up", "Min_ps_up", "Ave_ps_up",
                                    "Max_ps_do", "Min_ps_do", "Ave_ps_do",
                                    "Max_TCP_up", "Min_TCP_up", "Ave_TCP_up",
                                    "Max_TCP_do", "Min_TCP_do", "Ave_TCP_do",
                                    "Max_TTL_up", "Min_TTL_up", "Ave_TTL_up",
                                    "Max_TTL_do", "Min_TTL_do", "Ave_TTL_do",
                                    "FIN_up", "SYN_up", "RST_up", "PSH_up", "ACK_up", "URG_up",
                                    "FIN_do", "SYN_do", "RST_do", "PSH_do", "ACK_do", "URG_do",
                                    "chgcipher_up", "alert_up", "handshake_up", "appdata_up", "heartbeat_up",
                                    "chgcipher_do", "alert_do", "handshake_do", "appdata_do", "heartbeat_do",
                                    "Count", "Srv_count", "Same_src_rate", "Diff_src_rate", "Dst_host_count",
                                    "Dst_host_srv_count", "Dst_host_same_srv_rate", "Dst_host_diff_srv_rate"]

        self.__useful_columns_names = None

        # Initialization of the first general model
        self.gmm = GaussianMixture(n_components=self.k, covariance_type='diag', random_state=0)

        # Read initial training dataset and set it up for initial training process
        self.read_and_prepare_dataset()

        # Initial fit for avoiding non defined variable errors on first FL aggregation step
        self.gmm.fit(self.training_data)

    def read_and_prepare_dataset(self):
        # Read training data and convert to pandas dataframe
        self.training_data = pd.read_csv(self.training_file, delimiter=',')

        # Rename dataframe
        self.training_data.columns = self.__all_columns_names
        self.training_data[self.__all_columns_names[2:]] = self.training_data[self.__all_columns_names[2:]] \
            .astype(float)

        # Transform NA values to median
        self.training_data.fillna(self.training_data.median(), inplace=True)

        # Useless columns to remove
        useless_columns = ["Min_ps_up", "Min_ps_do", "Max_TCP_up", "Max_TCP_do",
                           "Min_TCP_do", "Max_TTL_up", "Min_TTL_up", "Ave_TTL_up", "Max_TTL_do",
                           "Min_TTL_do", "Ave_TTL_do", "RST_up", "URG_up", "RST_do", "ACK_do",
                           "URG_do", "heartbeat_do"]

        # Collect useful columns (all columns from __columns_names that are not marked as useless)
        self.__useful_columns_names = [col for col in self.__all_columns_names if col not in useless_columns]

        # Also ignore first 4 values (IP's and ports), as they are no longer useful
        self.__useful_columns_names = self.__useful_columns_names[4:]

        # Get rid of specified useless columns
        self.training_data.drop(useless_columns, axis=1, inplace=True)

        # Get rid of values with errors
        self.training_data = self.training_data[self.training_data["Duration"] >= 0].reset_index(drop=True)

        # Scale each feature value so it's between the range [0..1]
        scaler = MinMaxScaler()
        scaler.fit(self.training_data.iloc[:, 4:])

        # Finally, normalize final dataset
        self.training_data = pd.DataFrame(scaler.transform(self.training_data.iloc[:, 4:]))

    def accept_connections(self):
        connection, client_info = self.socket.accept()
        print(f"(INFO) New connection from {client_info}")
        return connection, client_info

    def send_message(self, action, connection, client_info):
        clear_msg = {"action": action, "model": self.model}
        try:
            encoded_msg = pickle.dumps(clear_msg)
            connection.sendall(encoded_msg)
        except BaseException as e:
            print(f"(EXCEPTION) Error decoding client {client_info} data: {e}")

    def send_for_training(self, client_info, connection):
        self.send_message("train", connection, client_info)
        print(f"(INFO) Model has been sent to client {client_info} for training")

    def send_definitive_model(self, client_info, connection):
        self.send_message("finished", connection, client_info)
        print(f"(INFO) Definitive model has been sent to client {client_info}")

    def surpasses_timeout(self, recv_start_time):
        return (time.time() - recv_start_time) > self.timeout

    @staticmethod
    def has_end_mark(data):
        return str(data)[-2] == '.'

    @staticmethod
    def is_an_update(data):
        return data["action"] == "update"

    def recv_update(self, tasks_queue, collected_responses):
        work = tasks_queue.get()
        index, connection, client_info = work[0], work[1], work[2]

        received_data = b''
        recv_start_time = time.time()
        done = False
        while not done:
            try:
                data = connection.recv(self.buffer_size)
                received_data += data

                if not data:  # Nothing received from the client.
                    received_data = b''
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute,
                    # return with status 0 to close the connection.
                    if self.surpasses_timeout(recv_start_time):
                        connection.close()
                        print(
                            f"(EXCEPTION) Connection closed with {client_info} either due to inactivity for "
                            f"{self.timeout} seconds or due to an error", end="\n\n")
                        collected_responses[index] = None
                        done = True

                elif self.has_end_mark(data):
                    print(
                        f"(INFO) All data ({len(received_data)} bytes) have been received from client {client_info}")
                    try:
                        # Decoding the data (bytes).
                        received_data = pickle.loads(received_data)
                        # Returning the decoded data.
                        collected_responses[index] = received_data
                        done = True

                    except BaseException as e:
                        print(f"(EXCEPTION) Error decoding client {client_info} data: {e}\n")
                        tasks_queue.task_done()
                        done = True

                else:
                    # In case data are received from the client, update the recv_start_time to the current time to
                    # reset the timeout counter.
                    recv_start_time = time.time()

            except BaseException as e:
                print(f"(EXCEPTION) Error receiving data from client {client_info}: {e}\n")
                done = True

        tasks_queue.task_done()

    @staticmethod
    def average(collection, total_samples):
        avg = []
        count = 0
        for update, n_training_samples in collection:
            update_weight = n_training_samples / total_samples
            if count == 0:
                avg = update
                for layer in range(len(avg)):
                    avg[layer] = avg[layer] * update_weight
            else:
                for layer in range(len(update)):
                    avg[layer] += update[layer] * update_weight
            count += 1

        return avg

    def average_updates(self, collected_updates):
        collected_weights, collected_biases = [], []
        total_samples = 0
        for update, n_training_samples in collected_updates:
            total_samples += n_training_samples
            collected_weights.append((update.coefs_, n_training_samples))
            collected_biases.append((update.intercepts_, n_training_samples))

        avg_weights = self.average(collected_weights, total_samples)
        avg_biases = self.average(collected_biases, total_samples)

        return avg_weights, avg_biases

    def print_results(self):
        score = self.model.score(inputs, expected_output)
        predictions = self.model.predict(inputs)
        print('(RESULTS) Score:', score)
        print('(RESULTS) Predictions:', predictions)
        print('(RESULTS) Expected:', np.array([0, 1, 1, 0]))
        print('(RESULTS) Accuracy: ', accuracy_score(np.array([0, 1, 1, 0]), predictions))

    def run(self):
        self.socket = socket.socket()
        print("(INFO) Socket is created")

        self.socket.bind((self.address, self.port))
        print("(INFO) Socket is bound to an address & port number")

        self.socket.listen(1)
        print("(INFO) Listening for incoming connections ...")

        # First phase: collecting clients
        print("(INFO) The federated training process is about to start. Collecting clients ... ")
        connected_clients = {}
        while len(connected_clients) < self.n_expected_clients:
            try:
                connection, client_info = self.accept_connections()
                connected_clients[client_info] = connection
            except socket.timeout:
                self.socket.close()
                print("(EXCEPTION) Socket closed because no connections received in a while")
                break

        print("(INFO) Collecting clients phase has ended (number of expected clients has been reached)")

        # Second phase: run rounds until convergence
        print("(INFO) The federated training process has started. Running rounds ...")
        rounds_completed = 0
        score = 0.0
        selected_clients = {}
        while (rounds_completed < self.rounds_limit) and (score < self.convergence_score):
            print(f"(INFO) Current round is {rounds_completed + 1} ")

            # Select C random fraction from all available clients
            n_selected_clients = len(connected_clients) * self.c_fraction
            selected = random.sample(list(connected_clients.items()), n_selected_clients)
            for sel in selected:
                selected_clients[sel[0]] = sel[1]

            # Send current model to all selected clients for update
            # Asynchronously wait for response from each client using a thread per client

            # Create queue for holding non-completed tasks (task = receive response (update) from 1 client)
            tasks_queue = Queue(maxsize=0)
            index = 0
            for client_info, connection in selected_clients.items():
                new_entry = (index, connection, client_info)
                tasks_queue.put(new_entry)
                index += 1

            # Create list for collecting responses from all clients. Each client's response received is appended to
            # this list
            collected_responses = [{} for client in selected_clients]

            for client_info, connection in selected_clients.items():
                self.send_for_training(client_info, connection)
                # Create one thread per client's response pending to allow parallel reception of the responses
                # Each client's thread gets a task from tasks_queue and appends its result to collected_responses
                client_thread = Thread(target=self.recv_update, args=(tasks_queue, collected_responses))
                client_thread.setDaemon(True)
                client_thread.start()

            # Wait until all updates have been received from all selected clients (no tasks remaining on the queue)
            tasks_queue.join()
            print("(INFO) All updates have been received from all selected clients")

            # Retrieve each update from each client's response and append it to collected_updates
            collected_updates = []
            for response in collected_responses:
                if response["action"] == "update":
                    collected_updates.append((response["model"], response["n_training_samples"]))

            # Average weights and biases from all updates received from selected clients
            avg_weights, avg_biases = self.average_updates(collected_updates)

            # Update server's model with averaged parameters
            self.model.coefs_ = avg_weights
            self.model.intercepts_ = avg_biases

            # Calculate updated model score
            score = self.model.score(inputs, expected_output)

            if score != 1.0:
                print(f"(INFO) Convergence hasn't been reached yet. Current score is {score}")

            # Update rounds counter
            rounds_completed += 1

        if rounds_completed >= self.rounds_limit:
            print(f"(INFO) Rounds limit ({self.rounds_limit}) has been reached and the model has not converged")
        if score == 1.0:
            print(f"(INFO) The model has been successfully trained in {rounds_completed} rounds")

        # Print definitive model's results
        self.print_results()

        # Send definitive model to connected clients
        print("(INFO) Proceeding to send the definitive model to all connected clients")
        for client_info, connection in connected_clients.items():
            self.send_definitive_model(client_info, connection)
        print("(INFO) Definitive model has been successfully sent to all connected clients")
        print("(INFO) The federated training process has ended")
        self.socket.close()
        print("(INFO) Socket has been closed")
        print("(INFO) Terminating execution ...")


if __name__ == '__main__':
    conf = ConfigFactory.parse_file('conf/server.conf')

    ADDRESS = conf.get_string('address')
    PORT = conf.get('port')
    BUFFER_SIZE = conf.get('buffer_size')
    TRAINING_FILE = conf.get('training_file')
    TIMEOUT = conf.get('timeout')
    ROUNDS_LIMIT = conf.get('rounds_limit')
    N_EXPECTED_CLIENTS = conf.get('n_expected_clients')
    C_FRACTION = conf.get('c_fraction')
    CONVERGENCE_SCORE = conf.get('convergence_score')

    server = FedAVGServer(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE, training_file=TRAINING_FILE,
                          timeout=TIMEOUT, rounds_limit=ROUNDS_LIMIT, n_expected_clients=N_EXPECTED_CLIENTS,
                          c_fraction=C_FRACTION, convergence_score=CONVERGENCE_SCORE)
    server.start()
