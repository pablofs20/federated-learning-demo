import socket
import pickle
import time
import threading
import numpy as np
from sklearn.neural_network import MLPClassifier

initial_inputs = np.array([[0, 0], [0, 1], [1,0], [1,1]])
initial_outputs = np.array([0, 0, 0, 0])

inputs = np.array([[0, 0], [0, 1], [1,0], [1,1]])
expected_output = np.array([0, 1, 1, 0])

nn = MLPClassifier(
    activation='logistic',
    max_iter=100,
    hidden_layer_sizes=(2,),
    solver='lbfgs')

nn.fit(initial_inputs, initial_outputs)

first_train = False

class SocketThread(threading.Thread):

    def __init__(self, connection, client_info, buffer_size, recv_timeout, lock):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout
        self.recv_start_time = time.time()
        self.lock = lock

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
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0  # 0 means the connection is no longer active and it should be closed.
                elif str(data)[-2] == '.':
                    print("All data ({data_len} bytes) have been received from {client_info}.".format(
                        client_info=self.client_info, data_len=len(received_data)))

                    if len(received_data) > 0:
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1

                        except BaseException as e:
                            print("Error decoding client {client_info} data: {error}.\n".format(
                                client_info=self.client_info, error=e))
                            return None, 0
                else:
                    # In case data are received from the client, update the recv_start_time to the current time to
                    # reset the timeout counter.
                    self.recv_start_time = time.time()

            except BaseException as e:
                print("Error receiving data from client {client_info}: {error}.\n".format(client_info=self.client_info,
                                                                                          error=e))
                return None, 0

    @staticmethod
    def proper_format(received_data):
        proper = False
        if (type(received_data) is dict) and ("data" in received_data.keys()) and ("action" in received_data.keys()):
            proper = True

        return proper



    def model_averaging(self, updated_model):
        global first_train
        if not first_train:
            updated_model_weights = updated_model.coefs_

            print(np.sum(updated_model_weights, nn.coefs_))

            print("COEFS ANTES")
            print(nn.coefs_)
            print(type(nn.coefs_))
            print("COEFS DEL UPDATED")
            print(updated_model_weights)
            print(type(nn.coefs_))
            nn.coefs_ = updated_model_weights
            print("COEFS DESPUES")
            print(nn.coefs_)
            print(type(nn.coefs_))
            first_train = True
        else:
            current_model_weights = nn.coefs_
            updated_model_weights = updated_model.coefs_
            #new_weights = self.average_coefs(current_model_weights, updated_model_weights)
            #nn.coefs_ = new_weights

    def reply(self, received_data):
        global nn
        if self.proper_format(received_data):
            action = received_data["action"]
            print("Client {client} required action: {action}".format(client=self.client_info, action=action))
            print(received_data)

            response = b''
            if action == "ready":  # send the model to the client for training
                try:
                    msg = {"action": "train", "data": nn}
                    response = pickle.dumps(msg)
                    self.connection.sendall(response)
                    print("Model have been sent to client for training")
                except BaseException as e:
                    print("Error decoding client {client_info} data: {error}".format(client_info=self.client_info,
                                                                                     error=e))

            elif action == "update":
                try:
                    new_model = received_data["data"]
                    self.model_averaging(new_model)
                    score = nn.score(inputs, expected_output)

                    if score < 0.8:
                        msg = {"action": "train", "data": nn}
                        response = pickle.dumps(msg)
                        self.connection.sendall(response)
                        print("Model have been sent to client for training 2")
                    else:
                        print("EUREKA ", score)
                        msg = {"action": "ok", "data": None}
                        response = pickle.dumps(msg)
                        self.connection.sendall(response)

                except BaseException as e:
                    print("Error decoding client {client_info} data: {error}".format(client_info=self.client_info,
                                                                                     error=e))

    def initialize_recv_timestamp(self):
        self.recv_start_time = time.time()
        time_struct = time.gmtime()
        date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(
            year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour,
            minute=time_struct.tm_min, second=time_struct.tm_sec)
        print(date_time)

    def run(self):
        while True:
            self.initialize_recv_timestamp()
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print(
                    "Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due "
                    "to an error.".format(
                        client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
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

    def accept_connections(self):
        connection, client_info = self.socket.accept()
        print("New connection from {client_info}.".format(client_info=client_info))
        lock = threading.RLock()  # for concurrent model modifications
        socket_thread = SocketThread(connection=connection,
                                     client_info=client_info,
                                     buffer_size=self.buffer_size,
                                     recv_timeout=self.timeout,
                                     lock=lock)
        socket_thread.start()

    def run(self):
        self.socket = socket.socket()
        print("Socket is created")

        self.socket.bind((self.address, self.port))
        print("Socket is bound to an address & port number")

        self.socket.listen(1)
        print("Listening for incoming connection ...")

        while True:
            try:
                self.accept_connections()
            except socket.timeout:
                self.socket.close()
                print("(Timeout) Socket closed because no connections received in a while")
                break


ADDRESS = "127.0.0.1"
PORT = 10003
BUFFER_SIZE = 1024
TIMEOUT = 10

if __name__ == '__main__':
    server = Server(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE, timeout=TIMEOUT)
    server.start()
