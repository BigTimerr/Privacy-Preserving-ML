import socket
import pickle
import struct

import numpy as np
import torch


class TCPServer(object):
    def __init__(self, addr, port, buf_size):
        self.address = (addr, port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.address)
        self.server_socket.listen(1)
        self.buf_size = buf_size
        self.client_socket = None
        self.buffer = None
        self.is_connected = False
        self.payload_size = struct.calcsize("L")
        self.data = b''
        self.party = 0

    def run(self):
        print("TCPServer waiting for connection ......")
        self.client_socket, client_address = self.server_socket.accept()
        self.is_connected = True
        print("TCPServer successfully connected by :%s" % str(client_address))

    def send_msg(self, msg):
        self.client_socket.send(msg)

    def send_value(self, v):
        if isinstance(v, int):
            package = np.array([v])
            self.send_np_array(package)
        else:
            self.send_np_array(v)

    def send_np_array(self, array):
        data = pickle.dumps(array)
        message_size = struct.pack("L", len(data))
        self.client_socket.sendall(message_size + data)

    def send_torch_array(self, array):
        array = array.cpu().detach().numpy()
        self.send_np_array(array)

    def receive_msg(self):
        recv_str = self.client_socket.recv(self.buf_size)
        return recv_str

    def receive_value(self):
        data = self.receive_np_array()
        if len(data) == 1:
            data = data[0]
        return data

    def receive_np_array(self):
        while len(self.data) < self.payload_size:
            self.data += self.client_socket.recv(self.buf_size)

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data += self.client_socket.recv(msg_size)
        self.client_socket.setblocking(1)
        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        return frame

    def receive_torch_array(self, device):
        frame = self.receive_np_array()
        frame = torch.from_numpy(frame).to(device)
        return frame

    def close(self):
        self.client_socket.close()


class TCPClient(object):
    def __init__(self, host, port, buf_size):
        self.address = (host, port)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        self.buf_size = buf_size
        self.server_socket = None
        self.buffer = None
        self.is_exit = False
        self.is_connected = False
        self.payload_size = struct.calcsize("L")
        self.data = b''
        self.party = 1

    def run(self):
        self.client_socket.connect(self.address)
        print('successfully connected to server: %s' % str(self.address[0]))
        self.is_connected = True

    def send_msg(self, msg):
        self.client_socket.send(msg)

    def send_value(self, v):
        if isinstance(v, int):
            package = np.array([v])
            self.send_np_array(package)
        else:
            self.send_np_array(v)

    def send_np_array(self, array):
        data = pickle.dumps(array)
        message_size = struct.pack("L", len(data))
        self.client_socket.sendall(message_size + data)

    def send_torch_array(self, array):
        array = array.cpu().detach().numpy()
        self.send_np_array(array)

    def receive_msg(self):
        recv_str = self.client_socket.recv(self.buf_size)
        return recv_str

    def receive_value(self):
        data = self.receive_np_array()
        if len(data) == 1:
            data = data[0]
        return data

    def receive_np_array(self):
        while len(self.data) < self.payload_size:
            self.data += self.client_socket.recv(self.buf_size)

        packed_msg_size = self.data[:self.payload_size]
        self.data = self.data[self.payload_size:]
        msg_size = struct.unpack("L", packed_msg_size)[0]

        # Retrieve all data based on message size
        while len(self.data) < msg_size:
            self.data += self.client_socket.recv(msg_size)
        frame_data = self.data[:msg_size]
        self.data = self.data[msg_size:]

        # Extract frame
        frame = pickle.loads(frame_data)
        return frame

    def receive_torch_array(self, device):
        frame = self.receive_np_array()
        frame = torch.from_numpy(frame).to(device)
        return frame

    def close(self):
        self.client_socket.close()
