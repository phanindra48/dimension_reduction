import numpy as np
import pandas as pd
import sys

def scatter (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  Y = np.genfromtxt(labelsPath)
  df = pd.DataFrame(X)
  X_mean = df.mean(axis=0)
  X_gmean = df.groupby(Y, axis=0).mean()

  M = df.groupby(Y, axis=0).count().to_numpy()[:,0]

  mean_diff = X_gmean - X_mean
  mean_diff_count = mean_diff.mul(M, axis=0)

  B = np.matmul(mean_diff_count.to_numpy().transpose(), mean_diff.to_numpy())

  # gives eigenvalues in asc order
  evals, evecs = np.linalg.eigh(B)
  print(evecs)
  # sort in reverse order
  idx = np.argsort(evals)[::-1]
  evals = evals[idx]
  evecs = evecs[:,idx]

  # get first k eigenvectors (least dominant) and set the rest of the columns to zero
  evecs[:,k:] = 0
  B_max = np.matmul(B, evecs)

  np.savetxt(vectorPath, evecs[:, :k], delimiter=',')
  np.savetxt(reducedDataPath, B_max[:, :k].transpose(), delimiter=',')
  return

if __name__ == "__main__":
  if len(sys.argv) < 5:
    print ('Please check parameters and try again.')
    print ('eg: python [program1] [input_data] [input_labels] [output_vector] [output_reduced_data]')
  else:
    dataPath = sys.argv[1]
    labelsPath = sys.argv[2]
    vectorPath = sys.argv[3]
    reducedDataPath = sys.argv[4]
    k = 2
    scatter(k, dataPath, labelsPath, vectorPath, reducedDataPath)
