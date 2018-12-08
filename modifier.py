import json

iF = open('data.json', 'r')
data = json.load(iF)


for key in list(data['albums'].keys()):
    if '$' in key:
        data['albums'].pop(key, None)


for key in list(data['clusters'].keys()):
    for x in range(len(data['clusters'][key]) - 1, -1, -1):
        if '$' in data['clusters'][key][x]:
            data['clusters'][key].pop(x)


json.dump(data, open('data2.json', 'w'))
