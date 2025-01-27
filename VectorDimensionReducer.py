import matplotlib.pyplot as plt
import numpy as np
import pprint

def DimensionalityReducer(matrix, k):
    # The final number is the number of points
    transposed = np.transpose(matrix)

    # Let's compute the M^T * M, and then find the eigenvectors for that
    M_final = np.matmul(transposed,matrix)
    
    # We get both the eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(M_final)
    
    # We want the k-th eigenvector, which will be the
    # dimension we want to reduce to.
    E = eigenvectors[:,k-1:]
    print(E)
    # Now we do ME (M is the original matrix), and we will get 
    # a dimensionality reduced matrix, that can be plotted.
    result = np.matmul(matrix,E)
    return result

def main():
    test_matrix1 = np.array([(1,2),(2,1),(3,4),(4,3)])
    M_reduced = DimensionalityReducer(test_matrix1,1)
    test_3D = np.random.randint(0,100,(10,3))
    M_reduced_3D = DimensionalityReducer(test_3D,2)
    print(test_3D)
    print(M_reduced_3D)
    plt.scatter(M_reduced_3D[:,0],M_reduced_3D[:,1],color="blue")
    plt.show()

if __name__ == "__main__":
    main()