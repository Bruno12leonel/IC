import pandas as pd
from river import stream
from sklearn.preprocessing import MinMaxScaler
from hdbstream import HDBStream
import sys

if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1], sep=',')

    scaler = MinMaxScaler()

    scaler.fit(data)

    data = pd.DataFrame(data=scaler.transform(data), columns=['x', 'y'])

    data = data.to_numpy()

    #denstream = CoreStream(m_minPoints= int(sys.argv[2]) , n_samples_init= int(sys.argv[3]), epsilon= float(sys.argv[4]))   

    hdbstream = HDBStream(200, step=2, decaying_factor=0.015, mu=2, n_samples_init=10000, epsilon = 0.006, stream_speed=100)

    count_points = 0
    
    for x, _ in stream.iter_array(data):
        _ = hdbstream.learn_one(x)
    
        count_points += 1
    
        if not (count_points % 10000) and count_points != 10000:
            hdbstream.predict_one()
