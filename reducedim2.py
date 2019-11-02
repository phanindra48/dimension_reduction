import numpy as np
import pandas as pd
import sys

# Same code as in scatter1.py
def scatter1 (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  Y = np.genfromtxt(labelsPath)
  df = pd.DataFrame(X)

  X_mean = (df - df.groupby(Y, axis=0).transform('mean')).to_numpy()
  W = np.matmul(X_mean.transpose(), X_mean)

  return W

# Same code as in scatter2.py
def scatter2 (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  Y = np.genfromtxt(labelsPath)
  df = pd.DataFrame(X)
  X_mean = df.mean(axis=0)
  X_gmean = df.groupby(Y, axis=0).mean()

  counts = df.groupby(Y, axis=0).count().to_numpy()[:,0]

  mean_diff = X_gmean - X_mean
  mean_diff_count = mean_diff.mul(counts, axis=0)

  B = np.matmul(mean_diff_count.to_numpy().transpose(), mean_diff.to_numpy())

  return B

def scatter3 (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  W = scatter1(k, dataPath, labelsPath, vectorPath, reducedDataPath)
  B = scatter2(k, dataPath, labelsPath, vectorPath, reducedDataPath)

  M = np.matmul(np.linalg.inv(W), B)

  # gives eigenvalues in asc order
  evals, evecs = np.linalg.eigh(M)
  # print(evecs)
  # sort in reverse order
  idx = np.argsort(evals)[::-1]
  evals = evals[idx]
  evecs = evecs[:,idx]

  # get first k eigenvectors (least dominant) and set the rest of the columns to zero
  evecs[:,k:] = 0
  M_max = np.matmul(M, evecs)

  np.savetxt(vectorPath, evecs[:, :k], delimiter=',')
  np.savetxt(reducedDataPath, M_max[:, :k].transpose(), delimiter=',')

  return


def reducedim (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  Y = np.genfromtxt(labelsPath)

  W = scatter1(k, dataPath, labelsPath, vectorPath, reducedDataPath)
  B = scatter2(k, dataPath, labelsPath, vectorPath, reducedDataPath)

  M = np.matmul(np.linalg.inv(W), B)
  # gives eigenvalues in asc order
  evals, evecs = np.linalg.eigh(M)
  # TODO: What to do when some of eigenvalues are negative
  # like how to get dominant eigenvectors
  print(evals)
  # sort in reverse order
  idx = np.argsort(evals)[::-1]
  evals = evals[idx]
  evecs = evecs[:,idx]

  D = np.matmul(X, evecs[:, :k])

  np.savetxt(vectorPath, evecs[:, :k], delimiter=',')
  np.savetxt(reducedDataPath, D.transpose(), delimiter=',')
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
    reducedim(k, dataPath, labelsPath, vectorPath, reducedDataPath)
