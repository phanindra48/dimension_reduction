import numpy as np
import sys

def pca (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  # Subtract mean
  X -= np.mean(X, axis=0)

  R = np.matmul(np.transpose(X), X)
  evals, evecs = np.linalg.eigh(R)
  print("evals=", evals)
  print("evecs=", evecs)
  # eigenvalues in increasing order, not decreasing order. Sort them.
  idx = np.argsort(evals)[::-1] # sort in reverse order
  evals = evals[idx]
  evecs = evecs[:,idx]
  print("evals=", evals)
  print("evecs=", evecs)
  # get first k eigenvectors
  V_k = evecs[:,:k]; print("V_k=",V_k)

  # store dominant k eighen vectors to file
  np.savetxt(vectorPath, V_k, delimiter=',')
  Mk = np.matmul(X, V_k)
  np.savetxt(reducedDataPath, Mk.transpose(), delimiter=',')
  return



if __name__ == "__main__":
  dataPath = 'iris.data'
  labelsPath = 'iris.labels'
  vectorPath = 'pca2_vectors.csv'
  reducedDataPath = 'pca2_reduced_data.csv'
  if False and len(sys.argv) < 5:
    print ('Please check parameters and try again.')
    print ('eg: python [program1] [input_data] [input_labels] [output_vector] [output_reduced_data]')
  else:
    dataPath = sys.argv[1] if len(sys.argv) > 1 else dataPath
    labelsPath = sys.argv[2] if len(sys.argv) > 2 else labelsPath
    vectorPath = sys.argv[3] if len(sys.argv) > 3 else vectorPath
    reducedDataPath = sys.argv[4] if len(sys.argv) > 4 else reducedDataPath
    k = 2
    pca(k, dataPath, labelsPath, vectorPath, reducedDataPath)
