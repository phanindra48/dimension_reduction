import numpy as np
import sys

def pca (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  R = np.matmul(np.transpose(X), X)
  evals, evecs = np.linalg.eigh(R)

  # sort in reverse order
  idx = np.argsort(evals)[::-1]
  evals = evals[idx]
  evecs = evecs[:,idx]

  # get first k eigenvectors
  V_k = evecs[:,:k]
  # print("V_k=",V_k)

  # store dominant k eighen vectors to file
  np.savetxt(vectorPath, V_k, delimiter=',')
  Mk = np.matmul(X, V_k)
  np.savetxt(reducedDataPath, Mk.transpose(), delimiter=',')
  return


if __name__ == "__main__":
  # dataPath = 'iris.data'
  # labelsPath = 'iris.labels'
  # vectorPath = 'pca1_vectors.csv'
  # reducedDataPath = 'pca1_reduced_data.csv'
  if len(sys.argv) < 5:
    print ('Please check parameters and try again.')
    print ('eg: python [program1] [input_data] [input_labels] [output_vector] [output_reduced_data]')
  else:
    dataPath = sys.argv[1]
    labelsPath = sys.argv[2]
    vectorPath = sys.argv[3]
    reducedDataPath = sys.argv[4]
    k = 2
    pca(k, dataPath, labelsPath, vectorPath, reducedDataPath)
