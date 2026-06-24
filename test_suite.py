import unittest
import read
from logical_classes import *
from function import KnowledgeBase


class KBTest(unittest.TestCase):

    def setUp(self):
        # Assert starter facts from statements.txt file
        file = 'statements.txt'
        self.data = read.read_tokenize(file)
        data = read.read_tokenize(file)
        self.KB = KnowledgeBase([], [])
        for item in data:
            if isinstance(item, Fact) or isinstance(item, Rule):
                self.KB.kb_assert(item)

        # Assert classification results, previously calculated,
        # as facts from classification_statements.txt file
        file_1 = 'classification_statements.txt'
        self.data = read.read_tokenize(file_1)
        data_1 = read.read_tokenize(file_1)
        for item in data_1:
            if isinstance(item, Fact) or isinstance(item, Rule):
                self.KB.kb_assert(item)

    # tests 1 through 4 are classification tasks
    # tests 5 through 12 sono basati sulla conoscenza del dominio diabete in statements.txt
    # tests 13 through 35 sono basati sui risultati della classificazione salvati in classification_statements.txt
    def test_5(self):
        ask1 = read.parse_input("fact: (isa diabetico ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : paziente")

    def test_6(self):
        ask1 = read.parse_input("fact: (inst paziente1 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : diabetico")

    def test_7(self):
        ask1 = read.parse_input("fact: (glucosio paziente1 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 148")

    def test_8(self):
        ask1 = read.parse_input("fact: (bmi paziente2 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 26.6")

    def test_9(self):
        ask1 = read.parse_input("fact: (eta paziente1 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 50")

    def test_10(self):
        ask1 = read.parse_input("fact: (pressione paziente2 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : 66")

    def test_11(self):
        ask1 = read.parse_input("fact: (inst paziente1 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : diabetico")

    def test_12(self):
        ask1 = read.parse_input("fact: (inst paziente2 ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        self.assertEqual(str(answer[0]), "?X : non_diabetico")

    def test_13(self):
        ask1 = read.parse_input("fact: (random_forest recall patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_14(self):
        ask1 = read.parse_input("fact: (naive_bayes auc patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_15(self):
        ask1 = read.parse_input("fact: (neural_network f1 patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_16(self):
        ask1 = read.parse_input("fact: (knn balanced_accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_17(self):
        ask1 = read.parse_input("fact: (logistic_regression precision patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_18(self):
        ask1 = read.parse_input("fact: (knn accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_19(self):
        ask1 = read.parse_input("fact: (decision_trees precision patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_20(self):
        ask1 = read.parse_input("fact: (logistic_regression f1 patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_21(self):
        ask1 = read.parse_input("fact: (k_means score patient ?X)")
        print(' Asking if', ask1)
        temp_values = self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))
        answer = self.KB.kb_ask(ask1)
        for i in range(0, 2):
            value = f"?X : {temp_values[i]}"
            self.assertEqual(str(answer[i]), value)

    def test_22(self):
        ask1 = read.parse_input("fact: (neural_network balanced_accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_23(self):
        ask1 = read.parse_input("fact: (random_forest auc patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_24(self):
        ask1 = read.parse_input("fact: (naive_bayes balanced_accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_25(self):
        ask1 = read.parse_input("fact: (knn precision patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_26(self):
        ask1 = read.parse_input("fact: (neural_network recall patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_27(self):
        ask1 = read.parse_input("fact: (logistic_regression accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_28(self):
        ask1 = read.parse_input("fact: (random_forest f1 patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_29(self):
        ask1 = read.parse_input("fact: (naive_bayes accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_30(self):
        ask1 = read.parse_input("fact: (neural_network auc patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_31(self):
        ask1 = read.parse_input("fact: (knn precision patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_32(self):
        ask1 = read.parse_input("fact: (k_means score patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        temp_values = self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))
        for i in range(0, len(temp_values)):
            value = f"?X : {temp_values[i]}"
            self.assertEqual(str(answer[i]), value)

    def test_33(self):
        r1 = read.parse_input("fact: (logistic_regression auc patient ?)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (logistic_regression balanced_accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_34(self):
        r1 = read.parse_input("fact: (decision_trees accuracy patient ?)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (decision_trees recall patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)

    def test_35(self):
        r1 = read.parse_input("fact: (neural_network auc patient ?)")
        print(' Retracting', r1)
        self.KB.kb_retract(r1)
        ask1 = read.parse_input("fact: (neural_network accuracy patient ?X)")
        print(' Asking if', ask1)
        answer = self.KB.kb_ask(ask1)
        value = f"?X : {self.KB.find_value(str(ask1.statement).replace('(', '').replace(')', '').split(' '))[0]}"
        self.assertEqual(str(answer[0]), value)


def pprint_justification(answer):
    """Pretty prints (hence pprint) justifications for the answer.
    """
    if not answer:
        print('Answer is False, no justification')
    else:
        print('\nJustification:')
        for i in range(0, len(answer.list_of_bindings)):
            # print bindings
            print(answer.list_of_bindings[i][0])
            # print justifications
            for fact_rule in answer.list_of_bindings[i][1]:
                pprint_support(fact_rule, 0)

def pprint_support(fact_rule, indent):
    """Recursive pretty printer helper to nicely indent
    """
    if fact_rule:
        print(' ' * indent, "Support for")

        if isinstance(fact_rule, Fact):
            print(fact_rule.statement)
        else:
            print(fact_rule.lhs, "->", fact_rule.rhs)

        if fact_rule.supported_by:
            for pair in fact_rule.supported_by:
                print(' ' * (indent + 1), "support option")
                for next in pair:
                    pprint_support(next, indent + 2)


if __name__ == '__main__':
    unittest.main()