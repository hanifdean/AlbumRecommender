from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import sys


dataF = open('dataset-7k-2012.txt', 'r')

data = []
d = {}

for line in dataF:
    features = line.rstrip().split('\t')
    album_id = features[0]
    d[album_id] = d.get(album_id, 0) + 1
    data.append(features)


for i in range(len(data) - 1, -1, -1):
    if(d.get(data[i][0]) != 12):
        data.pop(i)

data2 = []
album_ids = []
for idx, i in enumerate(data):
    if(len(album_ids) > 0 and album_ids[-1][0] == i[0]):
        for idx2, i2 in enumerate(i):
            if(idx2 > 3):
                data2[-1].append(i2)
    else:
        album_ids.append([])
        data2.append([])
        album_ids[-1].append(i[0])
        album_ids[-1].append(i[1])
        for idx2, i2 in enumerate(i):
            if(idx2 > 3):
                data2[-1].append(i2)


for i in range(len(data2)):
    new = []
    for j in range(len(data2[i]) - 13):
        new.append(float(data2[i][j+13]) - float(data2[i][j]))
    data2[i] = new

X = np.array(data2)

kmeans = KMeans(n_clusters=200).fit(X)

clusters = [ [] for x in range(200) ]

for i in range(len(data2)):
    clusters[kmeans.labels_[i]].append(album_ids[i])

print('\n')
second = False
m2 = []
n2 = []
o2 = []
fig, axs = plt.subplots(2)
for i in range(len(clusters)):
    print('Cluster %d:' % i)

    if(len(clusters[i]) == 3):
        m = []
        n = []
        o = []
        if(not second):
            for r in data:
                if(r[0] == clusters[i][0][0]):
                    m2.append(r[9])
                    print(r[9])
                elif(r[0] == clusters[i][1][0]):
                    n2.append(r[9])
                    print(r[9])
                elif(r[0] == clusters[i][2][0]):
                    o2.append(r[9])
                    print(r[9])
        else:
            for r in data:
                if(r[0] == clusters[i][0][0]):
                    m.append(r[9])
                    print(r[9])
                elif(r[0] == clusters[i][1][0]):
                    n.append(r[9])
                    print(r[9])
                elif(r[0] == clusters[i][2][0]):
                    o.append(r[9])
                    print(r[9])

        if(second):
            axs[0].plot([ t for t in range(12)], m,
                [ t for t in range(12) ], n,
                [ t for t in range(12) ], o)
            axs[0].set_xlabel('Track position [Cluster 1]')
            axs[0].set_ylabel('Feature')
            axs[1].plot([ t for t in range(12)], m2,
                [ t for t in range(12) ], n2,
                [ t for t in range(12) ], o2)
            axs[1].set_xlabel('Track position [Cluster 2]')
            axs[1].set_ylabel('Feature')
            plt.show()
            sys.exit()

        if(not second):
            second = True

    for j in clusters[i]:
        print(j[1])
    print('\n')
