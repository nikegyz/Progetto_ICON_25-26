import classification
from util import *
from classification import *
import os.path

verbose = 0

# Cache dei risultati di classificazione per dataset: la pipeline ML (e quindi
# l'apertura delle finestre dei grafici) viene eseguita UNA SOLA VOLTA per
# dataset, anche se più test pongono la stessa query di classificazione.
_classification_cache = {}


class KnowledgeBase(object):
    def __init__(self, facts=[], rules=[]):
        # Inizializza la Knowledge Base con fatti e regole
        self.facts = facts
        self.rules = rules
        self.ie = InferenceEngine()

    def __repr__(self):
        return 'KnowledgeBase({!r}, {!r})'.format(self.facts, self.rules)

    def __str__(self):
        string = "Knowledge Base: \n"
        string += "\n".join((str(fact) for fact in self.facts)) + "\n"
        string += "\n".join((str(rule) for rule in self.rules))
        return string

    def _get_fact(self, fact):
        """Uso interno: cerca un fatto uguale nella KB."""
        for kbfact in self.facts:
            if fact == kbfact:
                return kbfact

    def _get_rule(self, rule):
        """Uso interno: cerca una regola uguale nella KB."""
        for kbrule in self.rules:
            if rule == kbrule:
                return kbrule

    def kb_add(self, fact_rule):
        """Aggiunge un fatto o una regola alla KB."""
        printv("Adding {!r}", 1, verbose, [fact_rule])
        if isinstance(fact_rule, Fact):
            if fact_rule not in self.facts:
                self.facts.append(fact_rule)
                for rule in self.rules:
                    self.ie.fc_infer(fact_rule, rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.facts.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.facts[ind].supported_by.append(f)
                else:
                    ind = self.facts.index(fact_rule)
                    self.facts[ind].asserted = True
        elif isinstance(fact_rule, Rule):
            if fact_rule not in self.rules:
                self.rules.append(fact_rule)
                for fact in self.facts:
                    self.ie.fc_infer(fact, fact_rule, self)
            else:
                if fact_rule.supported_by:
                    ind = self.rules.index(fact_rule)
                    for f in fact_rule.supported_by:
                        self.rules[ind].supported_by.append(f)
                else:
                    ind = self.rules.index(fact_rule)
                    self.rules[ind].asserted = True

    def kb_assert(self, fact_rule):
        """Afferma un fatto o una regola nella KB."""
        printv("Asserting {!r}", 0, verbose, [fact_rule])
        self.kb_add(fact_rule)

    def kb_ask(self, fact):
        """Esegue una query sulla Knowledge Base."""
        print("Asking {!r}".format(fact))
        if factq(fact):
            if str(fact.statement).__contains__('classification'):
                results = self.ask_classification(fact)
                self.create_classification_statements(results)
                return [True]
            else:
                f = Fact(fact.statement)
                bindings_lst = ListOfBindings()
                for fact in self.facts:
                    binding = match(f.statement, fact.statement)
                    if binding:
                        bindings_lst.add_bindings(binding, [fact])
                return bindings_lst if bindings_lst.list_of_bindings else []
        else:
            print("Invalid ask:", fact.statement)
            return []

    def ask_classification(self, fact):
        """Seleziona il dataset giusto in base alla query di classificazione."""
        dataset = ''
        query = str(fact.statement).lower()
        
        # Route to Indian Liver Patient dataset for any classification query
        if 'patient' in query or 'liver' in query:
            dataset = './dataset/indian_liver_patient.csv'
        elif 'diabete' in query or 'diabetes' in query:
            # Legacy: if someone asks for diabetes, also use Indian Liver Patient
            dataset = './dataset/indian_liver_patient.csv'
        else:
            # Default: use Indian Liver Patient dataset
            dataset = './dataset/indian_liver_patient.csv'

        print(f"DEBUG ask_classification: query='{query}' -> dataset='{dataset}'")

        # Se la classificazione per questo dataset è già stata calcolata,
        # riusa il risultato (niente nuove finestre, niente ricalcolo).
        if dataset in _classification_cache:
            return _classification_cache[dataset]

        # Chiamata al modulo di classificazione (eseguita una sola volta)
        results = classification(dataset)
        _classification_cache[dataset] = results
        return results

    def create_file(self, results, type):
        """Crea un file di testo con le metriche del processo di classificazione."""
        path = './classification_file_' + type + '.txt'

        with open(path, 'w', encoding='utf-8') as classification_file:
            for x in results:
                if x.model != 'k_means':
                    new_line = (
                        f"{x.model}\n"
                        f"accuracy {x.accuracy}#f1 {x.f1}#precision {x.precision}#recall {x.recall}"
                        f"#balanced_accuracy {x.balanced_accuracy}#auc {x.auc}\n\n"
                    )
                else:
                    new_line = x.model + "\n"
                    if type == 'all':
                        k = 3
                    else:
                        k = 2
                    kmeans_lines = [line.strip() for line in str(x.kmeans_score).splitlines() if line.strip()]
                    for i in range(0, min(k, len(kmeans_lines))):
                        score_line = kmeans_lines[i]
                        if score_line.startswith('('):
                            score_line = score_line[1:]
                        try:
                            km_score = int(score_line)
                        except ValueError:
                            continue
                        new_line += str(km_score) + '\n'
                    new_line += '\n'

                classification_file.write(new_line)

    def create_classification_statements(self, results):
        """Scrive le dichiarazioni logiche basate sui risultati della classificazione."""
        path = './classification_statements.txt'
        file_type = ''

        for x in results:
            if 'indian_liver_patient' in str(x.calc_class).lower():
                file_type = 'patient'
            elif x.calc_class == './dataset/diabetes.csv':
                file_type = 'diabete'
            else:
                file_type = 'patient'  # Default to patient for Indian Liver Patient dataset

        with open(path, 'w', encoding='utf-8') as classification_statements:
            for x in results:
                if not str(x.model).__contains__('k_means'):
                    classification_statements.write(f"fact: ({x.model} accuracy {file_type} {x.accuracy})\n")
                    classification_statements.write(f"fact: ({x.model} f1 {file_type} {x.f1})\n")
                    classification_statements.write(f"fact: ({x.model} precision {file_type} {x.precision})\n")
                    classification_statements.write(f"fact: ({x.model} recall {file_type} {x.recall})\n")
                    classification_statements.write(f"fact: ({x.model} balanced_accuracy {file_type} {x.balanced_accuracy})\n")
                    classification_statements.write(f"fact: ({x.model} auc {file_type} {x.auc})\n")
                else:
                    if file_type == 'all':
                        k = 3
                    else:
                        k = 2
                    kmeans_lines = [line.strip() for line in str(x.kmeans_score).splitlines() if line.strip()]
                    for i in range(0, min(k, len(kmeans_lines))):
                        score_line = kmeans_lines[i]
                        if score_line.startswith('('):
                            score_line = score_line[1:]
                        try:
                            km_score = int(score_line)
                        except ValueError:
                            continue
                        classification_statements.write(f"fact: ({x.model} score {file_type} {km_score})\n")

        self.create_file(results, file_type)

    def find_value(self, fact):
        """Cerca un valore di metrica all'interno dei file di classificazione."""
        found_metric = []
        path = './classification_file_' + fact[2] + '.txt'

        if not os.path.isfile(path):
            return []

        with open(path, 'r', encoding='utf-8') as classification_file:
            lines = [line.strip() for line in classification_file]

        for index, line in enumerate(lines):
            if fact[0] in line:
                if fact[0] != 'k_means':
                    metric_parts = lines[index + 1].split('#')
                    for part in metric_parts:
                        key_value = part.strip().split(' ')
                        if len(key_value) == 2 and key_value[0] == fact[1]:
                            found_metric.append(key_value[1])
                else:
                    # k_means: leggi solo le righe numeriche realmente presenti
                    # (per 'patient' sono 2). Fermati al primo valore vuoto/non numerico
                    # per evitare disallineamenti con i binding restituiti dalla KB.
                    for k in range(0, 3):
                        idx = index + 1 + k
                        if idx >= len(lines) or not lines[idx]:
                            break
                        try:
                            int(lines[idx])
                        except ValueError:
                            break
                        found_metric.append(lines[idx])
                break

        return found_metric

    def kb_retract(self, fact_or_rule):
        """Rimuove un fatto o una regola dalla KB se non è supportato."""
        printv("Retracting {!r}", 0, verbose, [fact_or_rule])
        if len(fact_or_rule.supported_by) != 0:
            return None

        if isinstance(fact_or_rule, Rule):
            if fact_or_rule in self.rules and len(fact_or_rule.supported_by) == 0:
                self.rules.remove(fact_or_rule)

        if isinstance(fact_or_rule, Fact):
            flag = False
            for x in self.facts:
                if fact_or_rule.statement == x.statement:
                    fact_or_rule = x
                    flag = True
                    break
            if not flag:
                return None
            if len(fact_or_rule.supported_by) == 0:
                self.facts.remove(fact_or_rule)

        for temp in fact_or_rule.supports_facts:
            templen = 0
            standard = len(temp.supported_by)
            for x in temp.supported_by:
                if fact_or_rule in x:
                    temp.supported_by.remove(x)
                    templen += 1
            if standard == templen:
                self.kb_retract(temp)

        for temp in fact_or_rule.supports_rules:
            templen = 0
            standard = len(temp.supported_by)
            for y in temp.supported_by:
                if fact_or_rule in y:
                    temp.supported_by.remove(y)
                    templen += 1
            if standard == templen:
                self.kb_retract(temp)


class InferenceEngine(object):
    def fc_infer(self, fact, rule, kb):
        """Esegue inferenza forward chaining su un fatto e una regola."""
        printv('Attempting to infer from {!r} and {!r} => {!r}', 1, verbose,
               [fact.statement, rule.lhs, rule.rhs])
        bindings = match(rule.lhs[0], fact.statement)
        if bindings == False:
            return None
        if len(rule.lhs) == 1:
            newfact = Fact(instantiate(rule.rhs, bindings), [[rule, fact]])
            rule.supports_facts.append(newfact)
            fact.supports_facts.append(newfact)
            kb.kb_add(newfact)
        else:
            locallhs = []
            localrule = []
            for i in range(1, len(rule.lhs)):
                locallhs.append(instantiate(rule.lhs[i], bindings))
            localrule.append(locallhs)
            localrule.append(instantiate(rule.rhs, bindings))
            newrule = Rule(localrule, [[rule, fact]])
            rule.supports_rules.append(newrule)
            fact.supports_rules.append(newrule)
            kb.kb_add(newrule)
