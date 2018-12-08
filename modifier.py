import json

iF = open('data.json', 'r')
data = json.load(iF)

counter = 0

for key in list(data['albums'].keys()):
    if '$' in key:
        counter += 1
        print(counter)
        data.pop(key, None)

counter = 0

for key in list(data['clusters'].keys()):
    for x in range(len(data['clusters'][key]) - 1, -1, -1):
        if '$' in data['clusters'][key][x]:
            counter += 1
            print(counter)
            data['clusters'][key].pop(x)


json.dump(data, open('data2.json', 'w'))
