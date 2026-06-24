import unittest
import read
from function import KnowledgeBase


class KBTest(unittest.TestCase):

    def setUp(self):
        # In questa suite di test non è necessario aggiungere regole alla KB:
        # stiamo verificando solo che la query di classificazione venga gestita.
        self.KB = KnowledgeBase([], [])

    def test_classificazione_diabete(self):
        ask1 = read.parse_input("fact: (classification patient)")
        print('Calcolo', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)

    def test_classificazione_no_diabete(self):
        ask1 = read.parse_input("fact: (classification patient)")
        print('Calcolo', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)

    def test_classificazione_dataset_default(self):
        ask1 = read.parse_input("fact: (classification dataset)")
        print('Calcolo', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)

    def test_classificazione_metadati(self):
        ask1 = read.parse_input("fact: (classification patient)")
        print('Calcolo', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(answer[0], True)


if __name__ == '__main__':
    unittest.main()
