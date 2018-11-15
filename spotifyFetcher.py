import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sys
from time import sleep


YEAR = '2010'
TARGET_ALBUMS_NUM = 7000
OUTPUT_FILE = 'dataset-7k-2010.txt'
LOG_FILE = 'removed-2010.txt'


if(len(sys.argv) < 3):
    print("\n\n\n")
    print("Usage:")
    print("python spotifyFetcher.py <clientID> <clientSecret>")
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

oF = open(OUTPUT_FILE, 'w')
lF = open(LOG_FILE, 'w')

allAlbums = []


results = loopAndTry(sp.search, "CONNECTION REFUSED WHEN REQUESTING FOR ALBUMS", q='* year:' + YEAR, type='album')
while(results):
    print('Number of albums recorded:', len(allAlbums), end='\r')

    albums = results['albums']['items']

    for x in albums:
        album_name = x['name']
        album_id = x['id']
        album_tracks_num = x['total_tracks']

        if((album_name, album_id) not in allAlbums):
            allAlbums.append((album_name, album_id))

            r2 = loopAndTry(sp.album_tracks, "CONNECTION REFUSED WHEN REQUESTING FOR ALBUM TRACKS", album_id)
            while(r2):
                for track in r2['items']:
                    track_name = track['name']
                    track_id = track['id']

                    features = loopAndTry(sp.audio_features, "CONNECTION REFUSED WHEN REQUESTING FOR AUDIO FEATURES", [track_id])[0]
                    removed = False
                    if(features is None):
                        for r in range(10):
                            features = loopAndTry(sp.audio_features, "CONNECTION REFUSED WHEN REQUESTING FOR AUDIO FEATURES", [track_id])[0]
                            if(features is None):
                                if(r == 9):
                                    lF.write('%s\t%s\t%s\t%s\n' % (album_id, album_name, track_id, track_name))
                                    removed = True
                                continue
                            break

                    if(removed):
                        continue

                    del features['analysis_url']
                    del features['track_href']
                    del features['uri']
                    del features['type']
                    for key in features.keys():
                        if(features[key] is None):
                            lF.write('%s\t%s\t%s\t%s\n' % (album_id, album_name, track_id, track_name))
                            removed = True

                    if(removed):
                        continue

                    # [album id] [album name] [track id] [track name] [acousticness] [danceability] [duration (ms)] [energy] [instrumentalness] [key] [liveness] [loudness] [mode] [speechiness] [tempo] [time signature] [valence]
                    # delimiter = '\t'
                    oF.write('%s\t%s\t%s\t%s\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n' % (album_id, album_name, track_id, track_name,
                        features['acousticness'], features['danceability'], features['duration_ms'], features['energy'], features['instrumentalness'], features['key'],
                        features['liveness'], features['loudness'], features['mode'], features['speechiness'], features['tempo'], features['time_signature'], features['valence']))

                if(r2['next']):
                    r2 = loopAndTry(sp.next, "CONNECTION REFUSED WHEN REQUESTING FOR THE NEXT ALBUM TRACKS", r2)
                else:
                    r2 = None

            if(len(allAlbums) >= TARGET_ALBUMS_NUM):
                results['albums']['next'] = None
                break

    if(results['albums']['next']):
        results = loopAndTry(sp.next, "CONNECTION REFUSED WHEN REQUESTING FOR THE NEXT ALBUMS", results['albums'])
    else:
        results = None



print('Number of albums recorded:', len(allAlbums))
print('Finished fetching', TARGET_ALBUMS_NUM, 'albums in', YEAR)
lF.close()
oF.close()
