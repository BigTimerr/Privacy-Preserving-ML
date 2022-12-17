import TCP.tcp as tcp
from TCP.tcp import *
from utils_share_vector import calculate_c_i_next, int2bit_arr, getMSB

Ring = 10
n = 10
if __name__ == "__main__":
    client = TCPClient("127.0.0.1", 9999, 4096)
    client.run()

    share = np.random.randint(-Ring, Ring, size=n)
    share_arr = int2bit_arr(share, n)

    client.send_np_array(share_arr[:, 0])
    a1 = client.receive_np_array()

    c_i = np.asarray([False] * n)
    c_i_next = a1 & share_arr[:, 0]

    i = 0
    while i < 32:
        c_i = c_i_next
        b_i = share_arr[:, i]
        client.send_np_array(b_i)
        a_i = client.receive_np_array()
        c_i_next = calculate_c_i_next(a_i, b_i, c_i)
        i += 1

    b_h = share_arr[:, 31]
    client.send_np_array(b_h)
    a_h = client.receive_np_array()
    print(getMSB(a_h, b_h, c_i_next))