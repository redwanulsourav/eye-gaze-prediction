import socket
import threading

from ObservableDict import ObservableDict

predictedGaze = ObservableDict('predictions')
actualGaze = ObservableDict('ground truths')

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serverSocket.bind(('localhost', 12347))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clientSocket.bind(('localhost', 12348))

stopEvent = threading.Event()

def blockingClient():
    global clientSocket
    clientSocket.connect(('localhost', 12345))
    while not stopEvent.is_set():
        if predictedGaze.changed == True:
            predictedGaze.changed = False
            data = predictedGaze.data[0]
            data = data.ljust(1024, 'X')
            clientSocket.sendall(data.encode())
    try:
        clientSocket.shutdown(socket.SHUT_RDWR)
    except Exception as e:
        pass
    clientSocket.close()

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
                print(f'Received: {data}')
            except Exception as e:
                print(e)


if __name__ == '__main__':
    th0 = threading.Thread(target = blockingClient)
    th1 = threading.Thread(target = blockingServer)

    try:
        th0.start()
        th1.start()
        while True:
            data = input('>>> ')
            predictedGaze.updateKey(0, data)
    except KeyboardInterrupt:
        stopEvent.set()
        th0.join()
        th1.join()
        clientSocket.close()
        serverSocket.shutdown(socket.SHUT_RDWR)
        serverSocket.close()

    except Exception as e:
        stopEvent.set()
        clientSocket.close()
    finally:
        stopEvent.set()
        clientSocket.close()