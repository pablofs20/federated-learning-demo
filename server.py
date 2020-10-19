import socket
import pickle
import time
import threading
import numpy as np
from sklearn.neural_network import MLPClassifier

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([0, 1, 1, 0])

nn = MLPClassifier(
    activation='relu',
    max_iter=10000,
    hidden_layer_sizes=(4, 2),
    solver= 'lbfgs'
)

first_train = False
score = 0


class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size, recv_timeout, lock):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        self.recv_start_time = time.time()
        self.lock = lock

    def surpasses_timeout(self):
        return (time.time() - self.recv_start_time) > self.recv_timeout

    @staticmethod
    def has_end_mark(data):
        return str(data)[-2] == '.'

    def recv(self):
        received_data = b''
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if not data:  # Nothing received from the client.
                    received_data = b''
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute,
                    # return with status 0 to close the connection.
                    if self.surpasses_timeout:
                        return None, 0  # 0 means the connection is no longer active and it should be closed.
                elif self.has_end_mark(data):
                    print(
                        f"(INFO) All data ({len(received_data)} bytes) have been received from client {self.client_info}")

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print(f"(EXCEPTION) Error decoding client {self.client_info} data: {e}\n")
                            return None, 0
                else:
                    # In case data are received from the client, update the recv_start_time to the current time to
                    # reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print(f"(EXCEPTION) Error receiving data from client {self.client_info}: {e}\n")
                return None, 0

    @staticmethod
    def proper_format(received_data):
        proper = False
        if (type(received_data) is dict) and ("data" in received_data.keys()) and ("action" in received_data.keys()):
            proper = True

        return proper

    @staticmethod
    def model_averaging(updated_model):
        global first_train, nn
        if first_train:
            updated_model_weights = updated_model.coefs_
            current_model_weights = nn.coefs_
            updated_model_biases = updated_model.intercepts_
            current_model_biases = nn.intercepts_

            avg_weights = []
            for layer in range(0, len(updated_model_weights)):
                updated = updated_model_weights[layer]
                current = current_model_weights[layer]

                partial_result = np.array((updated + current) / 2)
                avg_weights.append(partial_result)

            avg_biases = []
            for layer in range(0, len(updated_model_biases)):
                updated = updated_model_biases[layer]
                current = current_model_biases[layer]

                partial_result = np.array((updated + current) / 2)
                avg_biases.append(partial_result)

            nn.coefs_ = avg_weights
            nn.intercepts_ = avg_biases
        else:
            nn = updated_model
            first_train = True

    def send_for_training(self):
        global nn
        msg = {"action": "train", "data": nn}
        try:
            response = pickle.dumps(msg)
            self.connection.sendall(response)
        except BaseException as e:
            print(f"(EXCEPTION) Error decoding client {self.client_info} data: {e}")

        print(f"(INFO) Model have been sent to client {self.client_info} for training")

    def send_trained_model(self):
        global nn
        msg = {"action": "finished", "data": nn}
        try:
            response = pickle.dumps(msg)
            self.connection.sendall(response)
        except BaseException as e:
            print(f"(EXCEPTION) Error decoding client {self.client_info} data: {e}")

    def reply(self, received_data):
        global nn, score
        if self.proper_format(received_data):
            action = received_data["action"]
            print(f"(INFO) Client {self.client_info} required action: {action}")

            if action == "ready":  # send the model to the client for training
                self.send_for_training()

            elif action == "update":  # update current model with new changes
                new_model = received_data["data"]

                with self.lock:
                    if score != 1.0:
                        self.model_averaging(new_model)
                        score = nn.score(inputs, expected_output)
                        predictions = nn.predict(inputs)
                        if score != 1.0:
                            print("(INFO) Convergence value has not been reached yet!")
                            self.send_for_training()
                        else:
                            self.send_trained_model()

                    else:
                        print("(INFO) Model has been successfully trained")
                        self.send_trained_model()

    def initialize_recv_timestamp(self):
        self.recv_start_time = time.time()
        time_struct = time.gmtime()
        date_time = f"(INFO) Waiting to Receive Data Starting from " \
                    f"{time_struct.tm_mday}/{time_struct.tm_mon}/{time_struct.tm_year} " \
                    f"{time_struct.tm_hour}:{time_struct.tm_min}:{time_struct.tm_sec} GMT "
        print(date_time)

    def run(self):
        while True:
            self.initialize_recv_timestamp()
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print(
                    f"(EXCEPTION) Connection closed with {self.client_info} either due to inactivity for "
                    f"{self.recv_timeout} seconds or due to an error", end="\n\n")
                break

            self.reply(received_data)


class Server(threading.Thread):
    def __init__(self, address, port, buffer_size, timeout):
        threading.Thread.__init__(self)
        self.address = address
        self.port = port
        self.buffer_size = buffer_size
        self.timeout = timeout
        self.socket = None

    def accept_connections(self, lock):
        connection, client_info = self.socket.accept()
        print(f"(INFO) New connection from {client_info}.")
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info,
                                     buffer_size=self.buffer_size,
                                     recv_timeout=self.timeout,
                                     lock=lock)
        socket_thread.start()

    def run(self):
        self.socket = socket.socket()
        print("(INFO) Socket is created")

        self.socket.bind((self.address, self.port))
        print("(INFO) Socket is bound to an address & port number")

        self.socket.listen(1)
        print("(INFO) Listening for incoming connection ...")

        lock = threading.RLock()  # for concurrent model modifications

        while True:
            try:
                self.accept_connections(lock)
            except socket.timeout:
                self.socket.close()
                print("(EXCEPTION) Socket closed because no connections received in a while")
                break


ADDRESS = "127.0.0.1"
PORT = 10003
BUFFER_SIZE = 200000
TIMEOUT = 10

if __name__ == '__main__':
    server = Server(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE, timeout=TIMEOUT)
    server.start()
