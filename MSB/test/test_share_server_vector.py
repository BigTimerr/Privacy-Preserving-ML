import TCP.tcp as tcp
from TCP.tcp import *
from utils_share_vector import calculate_c_i_next, int2bit_arr, getMSB

Ring = 10
n = 10
if __name__ == "__main__":
    server = TCPServer("127.0.0.1", 9999, 4096)
    server.run()

    share = np.random.randint(-Ring, Ring, size=n)
    share_arr = int2bit_arr(share, n)

    b1 = server.receive_np_array()
    server.send_np_array(share_arr[:, 0])
    c_i = np.asarray([False] * n)
    c_i_next = share_arr[:, 0] & b1

    i = 0
    while i < 32:
        c_i = c_i_next
        b_i = server.receive_np_array()
        a_i = share_arr[:, i]
        server.send_np_array(a_i)
        c_i_next = calculate_c_i_next(a_i, b_i, c_i)
        i +=1

    b_h = server.receive_np_array()
    a_h = share_arr[:, 31]
    server.send_np_array(a_h)
    print(getMSB(a_h, b_h, c_i_next))