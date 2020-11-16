import socket
import pickle
import time
import threading
import random
import numpy as np
from pyhocon import ConfigFactory
from queue import Queue
from threading import Thread
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

initial_inputs = np.array([[0, 0], [0, 1]])
initial_output = np.array([0, 1])

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 1, 1, 0])


class FedAVGServer(threading.Thread):
    def __init__(self, address, port, buffer_size, timeout, rounds_limit, n_expected_clients, c_fraction,
                 convergence_score):
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

        # initialization of the first general model
        self.model = MLPClassifier(activation='relu',
                                   max_iter=50000,
                                   hidden_layer_sizes=(4, 2),
                                   solver='sgd')

        # initial fit for avoiding non defined variable errors
        self.model.fit(initial_inputs, initial_output)

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
    TIMEOUT = conf.get('timeout')
    ROUNDS_LIMIT = conf.get('rounds_limit')
    N_EXPECTED_CLIENTS = conf.get('n_expected_clients')
    C_FRACTION = conf.get('c_fraction')
    CONVERGENCE_SCORE = conf.get('convergence_score')

    server = FedAVGServer(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE, timeout=TIMEOUT,
                          rounds_limit=ROUNDS_LIMIT, n_expected_clients=N_EXPECTED_CLIENTS, c_fraction=C_FRACTION,
                          convergence_score=CONVERGENCE_SCORE)
    server.start()
