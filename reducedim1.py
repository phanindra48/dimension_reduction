import numpy as np
import pandas as pd
import sys

def reducedim (k, dataPath, labelsPath, vectorPath, reducedDataPath):
  X = np.genfromtxt(dataPath, delimiter=',')
  Y = np.genfromtxt(labelsPath)

  R = np.matmul(X.transpose(), X)

  evals, evecs = np.linalg.eigh(R)

  # sort in reverse order
  idx = np.argsort(evals)[::-1]
  evals = evals[idx]
  evecs = evecs[:,idx]

  # get first k eigenvectors
  V_k = evecs[:,:k]

  D = np.matmul(X, V_k)

  np.savetxt(vectorPath, V_k, delimiter=',')
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
