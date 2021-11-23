
import pickle
import pandas as pd
from preprocess import process
from sklearn.linear_model import LogisticRegression

FILENAME = 'movies_IMDB.csv'

def main():
    model = pickle.loads(open('models/LogRegression_thre1'))
    process(filename='movies_IMDB.csv')
    df = pd.read_csv(FILENAME)
    df = df.drop(df.columns[[0]],axis=1)
    df = (df-df.mean())/(df.max()-df.min())
    X = np.array(df)
    predictions = model.predict(X)
    print predictions

if __name__ == '__main__':
    main()
