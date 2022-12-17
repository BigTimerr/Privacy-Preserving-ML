import tcp
import torch
import numpy as np
import time
import threading

# tcp_sen = TCPClient("127.0.0.1", 9998, 4096)
# tcp_rec = TCPServer("127.0.0.1", 9999, 4096)
#
# tcp_rec.run()
# tcp_sen.run()
#
#
# a = torch.rand(1000,2000)
#
# torch.save(a,"./myTensor.pt")
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
#
# s_t = time.time()
# tcp_sen.send_torch_array(a)
# other_a = tcp_sen.receive_torch_array()
# e_t = time.time()
# print(e_t - s_t)
#


tcp = tcp.TCPServer("127.0.0.1", 9999, 4096)
tcp.run()

a = tcp.receive_np_array()
print(a.shape)




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
#
# s_t = time.time()
# TCP.send_torch_array(a)
# other_a = TCP.receive_torch_array()
# e_t = time.time()
# print(e_t - s_t)