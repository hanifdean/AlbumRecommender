import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sys
# import pprint


if(len(sys.argv) < 3):
    print("\n\n\n")
    print("Usage:")
    print("python spotifyFetcher.py <clientID> <clientSecret>")
    sys.exit()


cred_manager = SpotifyClientCredentials(client_id=sys.argv[1], client_secret=sys.argv[2])
sp = spotipy.Spotify(client_credentials_manager=cred_manager)


MAX_LOOP = 1000
i = 0

albumsCount = 0
tracksCount = 0

while(i >= 0):
    results = sp.search(q='* year:2000', limit=50, offset=i*50+10000, type='album')
    albums = results['albums']['items']

    i += 1

    # pprint.pprint(results)

    if(len(albums) < 50):
        i = -1

    for x in albums:
        album_name = x['name']
        tracksNum = x['total_tracks']

        albumsCount += 1
        tracksCount += tracksNum

    print('Albums Count: ', albumsCount, '\tTracks Count: ', tracksCount)

    if(i > MAX_LOOP):
        i = -1
