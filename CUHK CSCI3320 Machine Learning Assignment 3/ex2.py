import os.path
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
# import imageio as misc  # noted that scipy.misc.imread is deprecated in SciPy 1.0.0, and will be removed in 1.2.0.
# import imageio as misc as alternative
from sklearn.decomposition import PCA

def load_data(digits = [0], num = 200):
    '''
    Loads all of the images into a data-array.

    The training data has 5000 images per digit,
    but loading that many images from the disk may take a while.  So, you can
    just use a subset of them, say 200 for training (otherwise it will take a
    long time to complete.

    Note that each image as a 28x28 grayscale image, loaded as an array and
    then reshaped into a single row-vector.

    Use the function display(row-vector) to visualize an image.

    '''
    totalsize = 0
    for digit in digits:
        totalsize += min([len(next(os.walk('train%d' % digit))[2]), num])
    print('We will load %d images' % totalsize)
    X = np.zeros((totalsize, 784), dtype = np.uint8)   #784=28*28
    for index in range(0, len(digits)):
        digit = digits[index]
        print('\nReading images of digit %d' % digit)
        for i in range(num):
            pth = os.path.join('train%d' % digit,'%05d.pgm' % i)
            image = misc.imread(pth).reshape((1, 784))
            X[i + index * num, :] = image
        print('\n')
    return X

def plot_mean_image(X, digits = [0]):
    ''' example on presenting vector as an image
    '''
    plt.close('all')
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.imshow(np.reshape(meanrow,(28,28)))
    plt.title('Mean image of digit ' + str(digits))
    plt.gray(), plt.xticks(()), plt.yticks(()), plt.show()

def main():
    digits = [0, 1, 2]
    # load handwritten images of digit 0, 1, 2 into a matrix X
    # for each digit, we just use 500 images
    # each row of matrix X represents an image
    X = load_data(digits, 500)
    # plot the mean image of these images!
    # you will learn how to represent a row vector as an image in this function
    plot_mean_image(X, digits)

    ####################################################################
    # plot the eigen images, eigenvalue v.s. the order of eigenvalue, POV
    # v.s. the order of eigenvalue
    # you need to
    #   1. do the PCA on matrix X;
    #
    #   2. plot the eigenimages (reshape the vector to 28*28 matrix then use
    #   the function ``imshow'' in pyplot), save the images of eigenvectors
    #   which correspond to largest 10 eigenvalues. Save them in a single file
    #   ``eigenimages.jpg''.
    #
    #   3. plot the POV (the Portion of variance explained v.s. the number of
    #   components we retain), save the figure in file ``digit_pov.jpg''
    #
    #   4. report how many dimensions are need to preserve 0.95 POV, describe
    #   your answers and your undestanding of the results in the plain text
    #   file ``description.txt''
    #
    #   5. remember to submit file ``eigenimages.jpg'', ``digit_pov.jpg'',
    #   ``description.txt'' and ``ex2.py''.
    # YOUR CODE HERE!

    ####################################################################

    pca = PCA()
    pca.fit(X)
    top10eigenvectors = pca.components_[:10]
    counter = 1
    for eigenvector in top10eigenvectors:
        plt.subplot(2, 5, counter)
        plt.imshow(eigenvector.reshape(28, 28))
        plt.gray(), plt.xticks(()), plt.yticks(()), plt.title("vector " + str(counter))
        counter += 1

    try:
        plt.savefig("eigenimages.jpg")
        # noted that matplotlib cannot save fig as jpg by default, installed Pillow for help
    except:
        plt.savefig("eigenimages.png")
    plt.show()
    plt.figure(figsize=(10, 10))

    PoV_cum = pca.explained_variance_ratio_.cumsum()
    plt.plot(range(1, len(PoV_cum) + 1), PoV_cum)
    plt.xlabel("Eigenvectors")
    plt.ylabel("Prop. of var.")
    try:
        plt.savefig("digit_pov.jpg")
        # noted that matplotlib cannot save fig as jpg by default, installed Pillow for help
    except:
        plt.savefig("digit_pov.png")
    plt.show()

    #     print(PoV_cum)
    #     print(PoV_cum[0:113])
    print("dimensions required to preserve 0.95 POV =", next(x[0] for x in enumerate(PoV_cum) if x[1] > 0.95) + 1)


if __name__ == '__main__':
    main()
