import matplotlib.pyplot as plt

iF = open('all.tsv', 'r')

data = {}
for line in iF:
    row = line.rstrip().split('\t')
    data[row[0]] = [ float(x) for x in row[1:] ]


snf = data['2oc73A20I1FTQiWGCkLeVP']
dw = data['1uyf3l2d4XYwiEqAb7t7fX']

vs = data['7fd7SEK25VS3gJAUgSwL6y']


fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)

danceability = 1

line1 = plt.plot([ x for x in range(0, 24) ], [ snf[x] for x in range(danceability, 13*24, 13) ], label='Science & Faith')
line2 = plt.plot([ x for x in range(0, 24) ], [ dw[x] for x in range(danceability, 13*24, 13) ], label='Doo-Wops & Hooligans')
line3 = plt.plot([ x for x in range(0, 24) ], [ vs[x] for x in range(danceability, 13*24, 13) ], label='Culture II (Other Cluster)')
plt.ylabel('Danceability')
plt.xlabel('Track Number')
plt.legend(loc='best')


ax2 = fig.add_subplot(1, 2, 2)
speechiness = 9

plt.plot([ x for x in range(0, 24) ], [ snf[x] for x in range(speechiness, 13*24, 13) ], label='Science & Faith')
plt.plot([ x for x in range(0, 24) ], [ dw[x] for x in range(speechiness, 13*24, 13) ], label='Doo-Wops & Hooligans')
plt.plot([ x for x in range(0, 24) ], [ vs[x] for x in range(speechiness, 13*24, 13) ], label='Culture II (Other Cluster)')
plt.ylabel('Speechiness')
plt.xlabel('Track Number')
fig.suptitle('Feature Comparison Between Albums from Different Clusters')
plt.legend(loc='best')
plt.show()
