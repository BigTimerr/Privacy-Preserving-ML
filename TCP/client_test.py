import tcp
import torch
import numpy as np
import threading
import time


# tcp_sen = TCPClient("127.0.0.1", 9999, 4096)
# tcp_rec = TCPServer("127.0.0.1", 9998, 4096)
#
# tcp_sen.run()
# tcp_rec.run()
#
# a = torch.rand(1000,2000)
#
#
# s_t = time.time()
# tcp_sen.send_torch_array(a)
# other_a = tcp_rec.receive_torch_array()
# e_t = time.time()
# print(e_t - s_t)
#
#
# print(other_a.shape)
#
# s_t = time.time()
# other_a = tcp_rec.receive_torch_array()
# tcp_rec.send_torch_array(a)
#
# e_t = time.time()
# print(e_t - s_t)


tcp = tcp.TCPClient("127.0.0.1", 9999, 4096)
tcp.run()

a = torch.rand(31,1,3,3)
print(a)
tcp.send_np_array(a)

# a = torch.rand(5000,1000)
#
# s_t = time.time()
# TCP.send_torch_array(a)
# other_a = TCP.receive_torch_array()
# e_t = time.time()
# print(e_t - s_t)
#
#
# print(other_a.shape)
#
#
#
# s_t = time.time()
# other_a = TCP.receive_torch_array()
# e_t = time.time()
# print(e_t - s_t)
# TCP.send_torch_array(a)
