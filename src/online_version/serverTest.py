import socket
import threading
import numpy as np
import json
from queue import Queue
from ObservableDict import ObservableDict
from model import CustomLSTM
import torch
import base64

predictedGaze = ObservableDict('predictions')
actualGaze = ObservableDict('ground truths')

serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
serverSocket.bind(('localhost', 12347))

clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
clientSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
clientSocket.bind(('localhost', 12348))

stopEvent = threading.Event()

recvQueue = Queue()
recvQueue2 = Queue()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomLSTM(64).to(device)
loss_fn = torch.nn.MSELoss()
optim = torch.optim.Adagrad(model.parameters(), lr=0.0001)

allFrames = None
allGazes = None

def blockingClient():
    global allFrames
    global allGazes
    try:
        while True:
            try:
                clientSocket.connect(('', 12345))
                break
            except BaseException as e:
                print('Client is not up yet')
                # pass
        print('Client is up')
        firstTime = True
        while not stopEvent.is_set():
            # print('Waiting')
            
            if recvQueue.empty() == False:
                data = recvQueue.get()
                # print(data)
                before, sep, after = data.partition('__')
                if sep == '':
                    recvQueue2.put(before)
                    continue
                # print(sep)
                data = ''
                while recvQueue2.empty() == False:
                    val = recvQueue2.get()
                    data += val
                    # print(val)
                data += before
                # print(data[-1])
                if after != '':
                    recvQueue2.put(after)
                # print(data)
                data = json.loads(data)
                
                frame = np.frombuffer(base64.b64decode(data['frame']), dtype=np.float16).reshape((224, 224, 3))
                # frame0 = frame.copy()
                frame0 = torch.from_numpy(frame)     # (224, 224, 3)
                frame0 = frame0.to(device)
                frame0 = frame0.permute(2, 0, 1)  # (3, 224, 224)
                frame0 = frame0.unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 224, 224)
                
                if allFrames == None:
                    allFrames = frame0
                else:
                    allFrames = torch.cat([allFrames, frame0], dim = 1)  # (1, T', 3, 224, 224)
                
                gaze = torch.Tensor(data['gaze'])   # (2)
                gaze = gaze.unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 2)

                if allGazes == None:
                    allGazes = gaze
                else:
                    allGazes = torch.cat([allGazes, gaze], dim = 1)
                
                if allGazes.shape[1] > 64:
                    print(f'dhukse')
                    model.train()
                    trainXFrame = allFrames[:, 0: -64, :, :, :].to(device)
                    print(f'trainXFrame: {trainXFrame.shape}')
                    trainXGaze = allGazes[:, 0: -64, :].to(device)
                    print(f'trainXGaze: {trainXGaze.shape}')
                    trainYGaze = allGazes[:, -64: , :].to(device)  # (1, 32, 2)
                    print(f'trainYGaze: {trainYGaze.shape}')
                    optim.zero_grad()

                    out = model(trainXFrame.float(), trainXGaze.float())
                    loss = loss_fn(out, trainYGaze)
                    print(f'loss: {loss.item()}')

                    loss.backward()
                    optim.step()

                    model.eval()
                    allFrames = allFrames.float().to(device)
                    allGazes = allGazes.float().to(device)
                    print(f'allFrames.shape: {allFrames.shape}')
                    print(f'allGazes.shape: {allGazes.shape}')
                    out = model(allFrames, allGazes)    # (1, 32, 2)
                    print(f'out hsape: {out.shape}')
                    # response = []
                    # for i in range(32):
                    #     response.append((allGazes.shape[1] + i + 1, out[0, i, 0].item(), out[0, i, 1].item()))
                    
                    tosend = {
                        'predicted': base64.b64encode(out.detach().numpy().tobytes()).decode(),
                        'startIdx': allFrames.shape[1]
                    }
                    data = json.dumps(tosend)
                    print(f'len: {data}')
                    data = data.ljust(4096, 'X')
                    print(f'sending: {data}')
                    clientSocket.sendall(data.encode())
            # if predictedGaze.changed == True:
                # predictedGaze.changed = False
                # data = predictedGaze.data[0]
                # data = data.ljust(1024, 'X')
                # clientSocket.sendall(data.encode())
        try:
            clientSocket.shutdown(socket.SHUT_RDWR)
        except Exception as e:
            pass
        clientSocket.close()
    except BaseException as e:
        print(e)

def blockingServer():
    firstTime = True
    serverSocket.listen()
    while not stopEvent.is_set():
        conn, addr = serverSocket.accept()
        while True:
            try:
                data = conn.recv(4096)
                if not data:
                    break
                if stopEvent.is_set():
                    break
                data = data.decode()
                
                recvQueue.put(data)
                # print(f'Received: {data}')
            except Exception as e:
                print(e)


if __name__ == '__main__':
    th0 = threading.Thread(target = blockingClient)
    th1 = threading.Thread(target = blockingServer)

    try:
        th0.start()
        th1.start()
        while True:
            pass

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