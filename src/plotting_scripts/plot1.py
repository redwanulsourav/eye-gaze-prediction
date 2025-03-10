import matplotlib.pyplot as plt
# import scienceplots
import os
import json

# plt.style.use(['ieee'])

if __name__ == '__main__':
    # Read GBVS and Itti Output

    f = open(f'../shreelock-gbvs/gbvs.log', 'r')
    contents = f.read()
    f.close()

    f = open(f'../runs/11/history.json')
    contents2 = f.read()
    f.close()

    data_dict = json.loads(contents2)

    contents = contents.split('\n')[: -2]

    frameCount = len(contents)

    gbvsValues = []
    ittiValues = []
    for line in contents:
        gbvs, itti = line.split(',')
        gbvsValue = float(gbvs.split('=')[-1])
        ittiValue = float(itti.split('=')[-1])

        gbvsValues.append(gbvsValue)
        ittiValues.append(ittiValue)
    
    plt.plot(range(0,frameCount), gbvsValues)
    plt.plot(range(0, frameCount), ittiValues)
    plt.plot(range(0, frameCount-1), data_dict["running_losses"])

    try:
        os.mkdir('plots')
    except Exception as e:
        pass
    
    plt.savefig('plots/plot1.jpg')

