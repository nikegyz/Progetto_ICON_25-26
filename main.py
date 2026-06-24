import sys
import os
import unittest



def main():
    import classification_test
    import test_suite

    print('\n----------------------------------------Executing Classification Tasks---------------------------------------')

    # executing the classification for all the datasets in the dataset folder
    # saving the results in classification_statements.txt
    suite1 = unittest.TestLoader().loadTestsFromModule(classification_test)
    unittest.TextTestRunner(verbosity=2).run(suite1)

    print('\n---------------------------------------------Executing Test Suite--------------------------------------------')

    # executing the test suite and setting up the KB with both statements.txt and classification_statements.txt files
    suite2 = unittest.TestLoader().loadTestsFromModule(test_suite)
    unittest.TextTestRunner(verbosity=2).run(suite2)

    print('\n--------------------------------------------Test Suite Completed---------------------------------------------')


if __name__ == '__main__':
    main()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
