import json

iF = open('data.json', 'r')
data = json.load(iF)


for key in list(data['albums'].keys()):
    if len(set('$#.[]/') & set(key)) > 0:
        new_key = key.replace('$', 's').replace('#', ' ').replace('.', ' ').replace('[', '(').replace(']', ')').replace('/', ' ')
        print(key)
        print(new_key)
        print('\n')
        data['albums'][new_key] = data['albums'].pop(key)


for key in list(data['clusters'].keys()):
    for x in range(len(data['clusters'][key]) - 1, -1, -1):
        if len(set('$#.[]/') & set(data['clusters'][key][x])) > 0:
            new_val = data['clusters'][key][x].replace('$', 'S').replace('#', ' ').replace('.', ' ').replace('[', '(').replace(']', ')').replace('/', ' ')
            data['clusters'][key][x] = new_val


json.dump(data, open('data2.json', 'w'))
