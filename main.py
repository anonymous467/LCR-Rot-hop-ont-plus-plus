# https://github.com/mtrusca/HAABSA_PLUS_PLUS
# https://github.com/ganeshjawahar/mem_absa
# https://github.com/Humanity123/MemNet_ABSA
# https://github.com/pcgreat/mem_absa
# https://github.com/NUSTM/ABSC

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from loadData import *

# import parameter configuration and data paths
from config import *

import lcrModelAlt_hierarchical_v1
import lcrModelAlt_hierarchical_v2
import lcrModelAlt_hierarchical_v3
import lcrModelAlt_hierarchical_v4

# main function
def main(_):
    loadData = False  # only for non-contextualised word embeddings.

    runLCRROTALT_v1 = False
    runLCRROTALT_v2 = False
    runLCRROTALT_v3 = False
    runLCRROTALT_v4 = True

    # determine if backupmethod is used
    if runLCRROTALT_v1 or runLCRROTALT_v2 or runLCRROTALT_v3 or runLCRROTALT_v4:
        backup = True
    else:
        backup = False

    # retrieve data and wordembeddings
    train_size, test_size, train_polarity_vector, test_polarity_vector = loadDataAndEmbeddings(FLAGS, loadData)
    print(test_size)
    remaining_size = 250
    accuracyOnt = 0.87

    test = FLAGS.test_path

    # LCR-Rot-hop model
    if runLCRROTALT_v1 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v1.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v2 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v2.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v3 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v3.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

    if runLCRROTALT_v4 == True:
        _, pred2, fw2, bw2, tl2, tr2 = lcrModelAlt_hierarchical_v4.main(FLAGS.train_path, test, accuracyOnt, test_size,
                                                                        remaining_size)
        tf.reset_default_graph()

print('Finished program succesfully')

if __name__ == '__main__':
    # wrapper that handles flag parsing and then dispatches the main
    tf.compat.v1.app.run()
