import socket
import time
import pickle
import threading
import numpy as np
from sklearn.metrics import accuracy_score

def_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
def_output = np.array([0, 1, 1, 0])


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
        print("(INFO) Socket is created")

        self.socket.connect((self.address, self.port))
        print("(INFO) Successfully connected to the server")

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
        print("(INFO) Training model received from server ...")
        self.model.fit(self.inputs, self.expected_output)
        print("(INFO) Model has been trained")

    def send_updated(self):
        msg = {"action": "update", "data": self.model}
        msg = pickle.dumps(msg)
        self.socket.sendall(msg)
        print("(INFO) Updated model has been sent to server")

    def test_and_print_results(self):
        score = self.model.score(def_inputs, def_output)
        predictions = self.model.predict(def_inputs)
        print('Score:', score)
        print('Predictions:', predictions)
        print('Expected:', np.array([0, 1, 1, 0]))
        print('Accuracy: ', accuracy_score(np.array([0, 1, 1, 0]), predictions))

        return score

    def send_confirmation(self):
        msg = {"action": "confirmation", "data": None}
        msg = pickle.dumps(msg)
        self.socket.sendall(msg)
        print("(INFO) Confirmation has been sent to server")

    def clear_buffer(self):
        self.received_data = b''

    def close_connection(self):
        self.socket.close()
        print("(INFO) Socket is closed")

    def run(self):
        self.initialize_connection()
        done = False
        cont = 0
        while not done:
            self.clear_buffer()
            self.receive_data()  # wait for server to send us the model

            if self.received_data["action"] == "train":
                self.parse_data()
                self.train_model()
                self.send_updated()

            if self.received_data["action"] == "finished":
                print("(INFO) Received definitive model from server")
                self.parse_data()
                print("(INFO) Testing the model ...")
                score = self.test_and_print_results()
                if score == 1.0:
                    print("(INFO) Testing OK")
                    self.close_connection()
                    done = 1
            cont += 1

        print("(INFO) Terminating execution ...")


ADDRESS = "127.0.0.1"
PORT = 10000
BUFFER_SIZE = 200000

if __name__ == '__main__':
    client = Client(address=ADDRESS, port=PORT, buffer_size=BUFFER_SIZE)
    client.start()
