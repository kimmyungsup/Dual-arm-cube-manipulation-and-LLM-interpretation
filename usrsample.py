import time, socket, threading, numpy as np

def motion_recv_task():
    global v
    rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rcv_sock.bind(('127.0.0.1', 6601))

    while True:
        data, clnt_addr = rcv_sock.recvfrom(1024)
        v = np.frombuffer(data, dtype=np.float32)
        #'v' contains
        # - dual-arm joint-pos(14)
        # - dual-arm joint-vel(14)
        # - dual-arm joint-current(14)
        # - left-hand joint pos(20)
        # - left-hand joint vel(20)
        # - left-hand joint current(20)
        # - right-hand joint pos(20)
        # - right-hand joint vel(20)
        # - right-hand joint current(20)
        # => total 162 * np.float32
        #'v' is global, so it can be read in the main thread (don't overwrite v in the main thread))

#start motion data receiving thread
v = np.zeros(162, dtype=np.float32) 
th = threading.Thread(target=motion_recv_task, daemon=True)
th.start()

#set command sending UDP socket
snd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
srv_addr = ('127.0.0.1', 6600)

#initialization motion (raising arms and make home pose)
cmd = 'init'
snd_sock.sendto(cmd.encode(), srv_addr)
time.sleep(3)
print(v)

#test joint motion
dual_arm_cmd   = 'joint -0.6 0.2 0.0 1.7 0.2 0.3 0.0  -0.6 0.2 0.0 1.7 0.2 0.3 0.0'
left_hand_cmd  = 'joint -0.5  1.0 -0.5 -0.5   0.2 0.5 0.5 0.5   0.1 0.5 0.5 0.5   0.0 0.5 0.5 0.5  -0.2 -0.2 0.5 0.5'
right_hand_cmd = 'joint  0.5 -1.0  0.5  0.5  -0.2 0.5 0.5 0.5  -0.1 0.5 0.5 0.5  -0.0 0.5 0.5 0.5   0.2  0.2 0.5 0.5' 
cmd = dual_arm_cmd + ', ' + left_hand_cmd + ', ' + right_hand_cmd 
snd_sock.sendto(cmd.encode(), srv_addr)
time.sleep(10.0)
print(v)

#test task motion
cmd = 'task  0.30 0.25 -0.40 0.0 0.0 0.0   0.30 -0.25 -0.40 0.0 0.0 0.0'
snd_sock.sendto(cmd.encode(), srv_addr)
time.sleep(10.0)
print(v)

#finishing motion (arms down)
cmd = 'rest'
snd_sock.sendto(cmd.encode(), srv_addr)
time.sleep(2.0)
print(v)

#shutdown the slave
cmd = 'quit'
snd_sock.sendto(cmd.encode(), srv_addr)
print(v)
