import numpy as np
import pandas as pd
import sys

def scatter (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  Y = np.genfromtxt(labelsPath)
  df = pd.DataFrame(X)

  X_mean = (df - df.groupby(Y, axis=0).transform('mean')).to_numpy()
  W = np.matmul(X_mean.transpose(), X_mean)
  # gives eigenvalues in asc order
  evals, evecs = np.linalg.eigh(W)
  # print("evals=", evals)
  print("evecs=", evecs)

  # get first k eigenvectors (least dominant) and set the rest of the columns to zero
  evecs[:,k:] = 0
  W_min = np.matmul(X_mean, evecs)

  np.savetxt(vectorPath, evecs[:, :k], delimiter=',')
  np.savetxt(reducedDataPath, W_min[:, :k].transpose(), delimiter=',')
  return

if __name__ == "__main__":
  dataPath = 'iris.data'
  labelsPath = 'iris.labels'
  vectorPath = 'scatter1_vectors.csv'
  reducedDataPath = 'scatter1_reduced_data.csv'
  if False and len(sys.argv) < 5:
    print ('Please check parameters and try again.')
    print ('eg: python [program1] [input_data] [input_labels] [output_vector] [output_reduced_data]')
  else:
    dataPath = sys.argv[1] if len(sys.argv) > 1 else dataPath
    labelsPath = sys.argv[2] if len(sys.argv) > 2 else labelsPath
    vectorPath = sys.argv[3] if len(sys.argv) > 3 else vectorPath
    reducedDataPath = sys.argv[4] if len(sys.argv) > 4 else reducedDataPath
    k = 2
    scatter(k, dataPath, labelsPath, vectorPath, reducedDataPath)
