import socket
import pickle
import time
import threading
import random
import numpy as np
from queue import Queue
from threading import Thread
from collections import OrderedDict
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

initial_inputs = np.array([[0, 0], [0, 1]])
initial_output = np.array([0, 1])

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 1, 1, 0])

class Server(threading.Thread):
    def __init__(self, address, port, buffer_size, timeout, rounds_limit):
        threading.Thread.__init__(self)
        self.address = address
        self.port = port
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.socket = None
        self.rounds_limit = rounds_limit
        self.connected_clients = {}
        self.selected_clients = {}
        self.n_connected_clients = 0
        self.n_expected_clients = 2
        self.n_selected_clients = 0
        self.c_fraction = 1  # select all available clients per round
        self.convergence_score = 1.0

        # initialization of the first general model
        self.model = MLPClassifier(activation='relu',
                                   max_iter=50000,
                                   hidden_layer_sizes=(4, 2),
                                   solver='lbfgs')

        # initial fit for avoiding non defined variable errors
        self.model.fit(initial_inputs, initial_output)

    def accept_connections(self, lock):
        connection, client_info = self.socket.accept()
        print(f"(INFO) New connection from {client_info}.")
        return connection, client_info

    def send_for_training(self, client_info, connection):
        clear_msg = {"action": "train", "data": self.model}
        try:
            encoded_msg = pickle.dumps(clear_msg)
            connection.sendall(encoded_msg)
        except BaseException as e:
            print(f"(EXCEPTION) Error decoding client {client_info} data: {e}")

        print(f"(INFO) Model have been sent to client {client_info} for training")

    def send_trained_model(self, client_info, connection):
        clear_msg = {"action": "finished", "data": self.model}
        try:
            encoded_msg = pickle.dumps(clear_msg)
            connection.sendall(encoded_msg)
        except BaseException as e:
            print(f"(EXCEPTION) Error decoding client {client_info} data: {e}")

        print(f"(INFO) Definitive model have been sent to client {client_info}")

    def surpasses_timeout(self, recv_start_time):
        return (time.time() - recv_start_time) > self.timeout

    @staticmethod
    def has_end_mark(data):
        return str(data)[-2] == '.'

    @staticmethod
    def is_an_update(data):
        return data["action"] == "update"

    def recv_update(self, tasks_queue, collected_responses):
        while not tasks_queue.empty():
            work = tasks_queue.get()
            index, connection, client_info = work[0], work[1], work[2]

            received_data = b''
            recv_start_time = time.time()
            while True:
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
                            tasks_queue.task_done()
                            break
                    elif self.has_end_mark(data):
                        print(
                            f"(INFO) All data ({len(received_data)} bytes) have been received from client {client_info}")

                        if len(received_data) > 0:
                            try:
                                # Decoding the data (bytes).
                                received_data = pickle.loads(received_data)
                                # Returning the decoded data.
                                collected_responses[index] = received_data
                                tasks_queue.task_done()
                                break

                            except BaseException as e:
                                print(f"(EXCEPTION) Error decoding client {client_info} data: {e}\n")
                                tasks_queue.task_done()
                                break
                    else:
                        # In case data are received from the client, update the recv_start_time to the current time to
                        # reset the timeout counter.
                        recv_start_time = time.time()

                except BaseException as e:
                    print(f"(EXCEPTION) Error receiving data from client {client_info}: {e}\n")
                    tasks_queue.task_done()
                    break

    def average(self, collection):
        avg = []
        count = 0
        for update in collection:
            if count == 0:
                avg = update
            else:
                for layer in range(len(update)):
                    avg[layer] += update[layer]
            count += 1

        for layer in range(len(avg)):
            avg[layer] = avg[layer] / 2

        return avg


    def average_updates(self, collected_updates):
        collected_weights, collected_biases = [], []
        for update in collected_updates:
            collected_weights.append(update.coefs_)
            collected_biases.append(update.intercepts_)

        avg_weights = self.average(collected_weights)
        avg_biases  = self.average(collected_biases)

        return avg_weights, avg_biases

    def run(self):
        self.socket = socket.socket()
        print("(INFO) Socket is created")

        self.socket.bind((self.address, self.port))
        print("(INFO) Socket is bound to an address & port number")

        self.socket.listen(1)
        print("(INFO) Listening for incoming connection ...")

        lock = threading.RLock()  # for concurrent model modifications

        # First phase: collecting clients
        while self.n_connected_clients < self.n_expected_clients:
            try:
                connection, client_info = self.accept_connections(lock)
                self.connected_clients[client_info] = connection
                self.n_connected_clients += 1
            except socket.timeout:
                self.socket.close()
                print("(EXCEPTION) Socket closed because no connections received in a while")
                break

        # Second phase: run rounds until convergence
        rounds_completed = 0
        score = 0.0
        while (rounds_completed < self.rounds_limit) and (score < self.convergence_score):
            self.n_selected_clients = self.n_connected_clients * self.c_fraction

            # Select C random fraction from all available clients
            selected = random.sample(list(self.connected_clients.items()), 2)
            for sel in selected:
                self.selected_clients[sel[0]] = sel[1]

            # Send current model to all selected clients for update
            # Create queue for holding non-completed tasks (task = receive response (update) from 1 client)
            tasks_queue = Queue(maxsize=0)
            index = 0
            for client_info, connection in self.selected_clients.items():
                new_entry = (index, connection, client_info)
                tasks_queue.put(new_entry)
                index += 1

            # Create list for managing responses from all clients. Each client's response received is appended to
            # this list
            collected_responses = [{} for client in self.selected_clients]  # For collecting responses from each client

            for client_info, connection in self.selected_clients.items():
                self.send_for_training(client_info, connection)
                # Create one thread per client's response pending to allow parallel reception of the responses
                # Each client's thread gets a task from tasks_queue and appends its result to collected_responses
                client_thread = Thread(target=self.recv_update, args=(tasks_queue, collected_responses))
                client_thread.setDaemon(True)
                client_thread.start()

            # Check if all updates have been received from all selected clients (no tasks remaining on the queue)
            tasks_queue.join()
            print("(INFO) All data has been received from all selected clients")

            # Retrieve each update from each client's response and append it to collected_updates
            collected_updates = []
            for response in collected_responses:
                if response["action"] == "update":
                    collected_updates.append(response["data"])

            # Average weights and biases from all updates received from selected clients
            avg_weights, avg_biases = self.average_updates(collected_updates)

            # Update server's model with averaged parameters
            self.model.coefs_ = avg_weights
            self.model.intercepts_ = avg_biases

            print("(INFO) Model has been successfully averaged")

            # Calculate updated model score
            score = self.model.score(inputs, expected_output)

            # Update rounds counter
            rounds_completed += 1

        if rounds_completed >= self.rounds_limit:
            print("(INFO) Rounds limit has been reached and the model hasn't converged")
        if score == 1.0:
            print(f"(INFO) The model has been succesfully trained in {rounds_completed} rounds")
            score = self.model.score(inputs, expected_output)
            predictions = self.model.predict(inputs)
            print('Score:', score)
            print('Predictions:', predictions)
            print('Expected:', np.array([0, 1, 1, 0]))
            print('Accuracy: ', accuracy_score(np.array([0, 1, 1, 0]), predictions))
            # Send definitive model to connected clients
            print("(INFO) Proceeding to send the definitive model to all connected clients")
            for client_info, connection in self.connected_clients.items():
                self.send_trained_model(client_info, connection)
            print("(INFO) Definitive model has been successfully sent to all connected clients")
            self.socket.close()
            print("(INFO) Socket has been closed")
            print("(INFO) Terminating execution ...")


ADDRESS = "127.0.0.1"
PORT = 10000
BUFFER_SIZE = 200000
TIMEOUT = 10
ROUNDS_LIMIT = 50000

if __name__ == '__main__':
    server = Server(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE, timeout=TIMEOUT, rounds_limit=ROUNDS_LIMIT)
    server.start()
