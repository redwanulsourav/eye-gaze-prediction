import socket
import threading
from queue import Queue

from ObservableDict import ObservableDict

predictedGaze = ObservableDict('predictions')
actualGaze = ObservableDict('ground truths')

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serverSocket.bind(('', 12345))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clientSocket.bind(('localhost', 12346))

dataQ = Queue()

stopEvent = threading.Event()    

def blockingServer():
    serverSocket.listen()
    while not stopEvent.is_set():
        conn, addr = serverSocket.accept()
        while True:
            try:
                data = conn.recv(1024)
                if not data:
                    break
                if stopEvent.is_set():
                    break
                data = data.decode().rstrip('X')
                dataQ.put(data)
                print(f'Received: {data}')
            except Exception as e:
                print(e)

def blockingClient():
    while True:
        try:
            clientSocket.connect(('localhost', 12347))
            print('Connected with server')
            break
        except Exception as e:
            print('Server is not up yet')
        
    while not stopEvent.is_set():
        if dataQ.empty() == False:
            # predictedGaze.changed = False
            data = dataQ.get() + '_P'
            data = data.ljust(1024, 'X')
            clientSocket.sendall(data.encode())
    try:
        clientSocket.shutdown(socket.SHUT_RDWR)
    except Exception as e:
        pass
    clientSocket.close()

def mediaPlayer():
    pass

if __name__ == '__main__':
    th0 = threading.Thread(target = blockingServer)
    th1 = threading.Thread(target = blockingClient)
    try:
        th0.start()
        th1.start()
        while True:
            pass
    except:
        stopEvent.set()
        th0.join()
        th1.join()
        serverSocket.shutdown(socket.SHUT_RDWR)
        serverSocket.close()

    
        