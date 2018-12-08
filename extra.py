import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sys
from time import sleep
import json
import pprint


OUTPUT_FILE = 'results/withAlbumAsKey.json'
INPUT_FILE = 'results/withAlbumAsKey.tsv'
OUTPUT_FILE2 = 'results/withClusterAsKey.json'
INPUT_FILE2 = 'results/withClusterAsKey.tsv'
OUTPUT_FILE3 = 'data.json'


if(len(sys.argv) < 3):
    print("\n\n\n")
    print("Usage:")
    print("python extra.py <clientID> <clientSecret>")
    sys.exit()



def loopAndTry(func, debug, *args, **kwargs):

    loop = 1

    while(loop <= 10):
        try:
            return func(*args, **kwargs)
        except:
            print(debug)
            print("RETRYING AFTER %d MINUTES" % (loop * 5))
            sleep(loop * 300)
            loop += 1

    print("ERROR REQUESTING THE DATA FOR 10 TIMES")
    sys.exit()



cred_manager = SpotifyClientCredentials(client_id=sys.argv[1], client_secret=sys.argv[2])
sp = spotipy.Spotify(client_credentials_manager=cred_manager)

iF = open(INPUT_FILE, 'r')
oF = open(OUTPUT_FILE, 'w')

albums_dict = {}
counter = 0

for line in iF:
    if counter % 500 == 0:
        print("Finished %d albums" % counter)
    row = line.rstrip().split('\t')
    # Make [ album_id, cluster_id ]
    row = [ row[0].split('_')[-1], row[1] ]
    album = loopAndTry(sp.album, "CONNECTION REFUSED WHEN REQUESTING FOR AN ALBUM", row[0])
    album_name = album['name']
    artist_name = album['artists'][0]['name']
    url = album['external_urls']['spotify']
    try:
        img_url = album['images'][0]['url']

        albums_dict[album_name + '_' + row[0]] = {
            'cluster_id': row[1],
            'artist': artist_name,
            'url': url,
            'img_url': img_url
        }
    except:
        albums_dict[album_name + '_' + row[0]] = {
            'cluster_id': row[1],
            'artist': artist_name,
            'url': url,
            'img_url': False
        }
    counter += 1

json.dump(albums_dict, oF)
oF.close()
iF.close()

print('Finished writing withAlbumAsKey.tsv to withAlbumAsKey.json')


iF = open(INPUT_FILE2, 'r')
oF = open(OUTPUT_FILE2, 'w')

clusters_dict = {}

for line in iF:
    row = line.rstrip().split('\t')
    clusters_dict[row[0]] = [ AN_AI for AN_AI in row[1:] ]

json.dump(clusters_dict, oF)
oF.close()
iF.close()

print('Finished writing withClusterAsKey.tsv to withClusterAsKey.json')


oF = open(OUTPUT_FILE3, 'w')

final_dict = {
    'albums': albums_dict,
    'clusters': clusters_dict
}

json.dump(final_dict, oF)

print('Finished all')
