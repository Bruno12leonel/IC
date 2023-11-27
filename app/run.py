import pandas as pd
from river import stream
from sklearn.preprocessing import MinMaxScaler
from corestream import CoreStream
import sys

if __name__ == "__main__":

    data = pd.read_csv(sys.argv[1], sep=',')

    #variável de pontos iniciais
    initial_points = int(sys.argv[2])

    scaler = MinMaxScaler()

    scaler.fit(data)

    data = pd.DataFrame(data=scaler.transform(data))

    data = data.to_numpy()

    #denstream = CoreStream(m_minPoints= int(sys.argv[2]) , n_samples_init= int(sys.argv[3]), epsilon= float(sys.argv[4]))   

    corestream = CoreStream(int(sys.argv[3]),
                        min_cluster_size = 25,
                        step=2,
                        decaying_factor=0.025,
                        mu=2, n_samples_init=initial_points, 
                        epsilon=0.005,
                        percent=0.15,
                        method_summarization='single',
                        stream_speed=100,
                        runtime=False,
                        plot=False,
                        save_partitions=True)

    count_points = 0

    for x, _ in stream.iter_array(data):
        denstream = corestream.learn_one(x)
        
        count_points += 1
        
        if not (count_points % initial_points) and count_points != initial_points:
            corestream.predict_one()
            #corestream.save_runtime_timestamp()
    #corestream.save_runtime_final()
