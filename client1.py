import socket
import pickle
import threading
import numpy as np
from sklearn.neural_network import MLPClassifier


class Client(threading.Thread):
    def __init__(self, address, port, buffer_size):
        threading.Thread.__init__(self)
        self.address = address
        self.port = port
        self.buffer_size = buffer_size
        self.socket = None
        self.received_data = b''
        self.action = None
        self.model = None
        self.inputs = np.array([[0, 0], [0, 1]])
        self.expected_output = np.array([0, 1])

    def initialize_connection(self):
        self.socket = socket.socket()
        print("Socket is created")

        self.socket.connect((self.address, self.port))
        print("Successfully connected to the server")

    def solicit_model(self):
        msg = {"action": "ready", "data": None}
        msg = pickle.dumps(msg)
        self.socket.sendall(msg)

    def receive_data(self):
        while str(self.received_data)[-2] != '.':
            data = self.socket.recv(self.buffer_size)
            self.received_data += data

        self.received_data = pickle.loads(self.received_data)

    def parse_data(self):
        if (type(self.received_data) is dict) and ("data" in self.received_data.keys()) and (
                "action" in self.received_data.keys()):
            self.action = self.received_data["action"]
            self.model = self.received_data["data"]

    def train_model(self):
        self.model.fit(self.inputs, self.expected_output)

    def send_updated(self):
        msg = {"action": "update", "data": self.model}
        msg = pickle.dumps(msg)
        self.socket.sendall(msg)

    def clear_buffer(self):
        self.received_data = b''

    def close_connection(self):
        self.socket.close()
        print("Socket is closed")

    def run(self):
        self.initialize_connection()
        self.solicit_model()

        done = 0
        while not done:
            self.clear_buffer()
            self.receive_data()
            self.parse_data()

            if self.action == "train":
                self.train_model()
                self.send_updated()
            else:
                # Nothing to do, receive model, send confirmation to server and close connection
                print(self.received_data)
                self.close_connection()
                done = 1


ADDRESS = "127.0.0.1"
PORT = 10003
BUFFER_SIZE = 1024

if __name__ == '__main__':
    client = Client(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE)
    client.start()
