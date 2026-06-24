# NB: 'is_var' viene importato in modo pigro (dentro i metodi che lo usano)
# per evitare l'import circolare util <-> logical_classes.


class Fact(object):
    """Rappresenta un fatto nella knowledge base. Contiene una dichiarazione
        come (inst paziente1 diabetico) e tiene traccia di quali fatti o regole
        lo supportano e quali fatti o regole supporta.
    Attributi:
        name (str): 'fact', il nome di questa classe
        statement (Statement): la dichiarazione contenuta nel fatto
        asserted (bool): se il fatto è dichiarato esplicitamente o inferito
        supported_by (listof Fact|Rule): fatti/regole che supportano questo fatto
        supports_facts (listof Fact): fatti che questo fatto supporta
        supports_rules (listof Rule): regole che questo fatto supporta
    """

    def __init__(self, statement, supported_by=[]):
        """Costruisce un fatto impostando i campi utili e generando la dichiarazione.
        Args:
            statement (str|Statement): La dichiarazione del fatto, cioè ciò che il fatto esprime
            supported_by (listof Fact|Rule): Fatti/Regole che consentono l'inferenza di questo fatto
        """
        super(Fact, self).__init__()
        self.name = "fact"
        self.statement = statement if isinstance(statement, Statement) else Statement(statement)
        self.asserted = not supported_by
        # self.supported_by = supported_by
        self.supported_by = []
        self.supports_facts = []
        self.supports_rules = []
        for pair in supported_by:
            self.supported_by.append(pair)

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Fact({!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(
            self.name, self.statement,
            self.asserted, self.supported_by,
            self.supports_facts, self.supports_rules)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        string = self.name + ":\n"
        string += "\t" + str(self.statement) + "\n"
        string += "\t Asserted:       " + str(self.asserted) + "\n"
        if self.supported_by != []:
            name_strings = [str(x.name) for y in self.supported_by for x in y]
            supported_by_str = ", ".join(name_strings)
            string += "\t Supported by:   [" + supported_by_str + "]\n"
        if self.supports_facts != []:
            name_strings = [str(x.name) for x in self.supports_facts]
            supports_f_str = ", ".join(name_strings)
            string += "\t Supports facts: [" + supports_f_str + "]\n"
        if self.supports_rules != []:
            name_strings = [str(x.name) for x in self.supports_rules]
            supports_r_str = ", ".join(name_strings)
            string += "\t Supports rules: [" + supports_r_str + "]\n"
        return string

    def __eq__(self, other):
        # Definisce il comportamento di == per questo oggetto
        return isinstance(other, Fact) and self.statement == other.statement

    def __ne__(self, other):
        # Definisce il comportamento di != per questo oggetto
        return not self == other


class Rule(object):
    """Rappresenta una regola nella knowledge base. Ha un elenco di dichiarazioni
        LHS che devono essere vere per inferire la dichiarazione RHS.
        Tiene traccia anche dei fatti e delle regole che la supportano e di quelli
        che essa stessa supporta.
    Attributi:
        name (str): 'rule', il nome di questa classe
        lhs (listof Statement): dichiarazioni sulla sinistra della regola
        rhs (Statement): dichiarazione inferita sulla destra della regola
        asserted (bool): se la regola è dichiarata esplicitamente o inferita
        supported_by (listof Fact|Rule): fatti/regole che permettono l'inferenza
        supports_facts (listof Fact): fatti supportati da questa regola
        supports_rules (listof Rule): regole supportate da questa regola
    """

    def __init__(self, rule, supported_by=[]):
        """Costruisce una regola impostando i campi utili e generando il LHS e il RHS.
        Args:
            rule (listof list): Rappresentazione grezza delle dichiarazioni che compongono LHS e RHS
            supported_by (listof Fact|Rule): Fatti/Regole che consentono l'inferenza di questa regola
        """
        super(Rule, self).__init__()
        self.name = "rule"
        self.lhs = [statement if isinstance(statement, Statement) else Statement(statement) for statement in rule[0]]
        self.rhs = rule[1] if isinstance(rule[1], Statement) else Statement(rule[1])
        self.asserted = not supported_by
        self.supported_by = []
        self.supports_facts = []
        self.supports_rules = []
        for pair in supported_by:
            self.supported_by.append(pair)

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Rule({!r}, {!r}, {!r}, {!r}, {!r}, {!r}, {!r})'.format(
            self.name, self.lhs, self.rhs,
            self.asserted, self.supported_by,
            self.supports_facts, self.supports_rules)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        string = self.name + ":\n"
        string += "\t Left hand:\n"
        for statement in self.lhs:
            string += "\t\t" + str(statement) + "\n"
        string += "\t Right hand:\n\t\t" + str(self.rhs) + "\n"
        string += "\t Asserted:       " + str(self.asserted) + "\n"
        if self.supported_by != []:
            name_strings = [str(x.name) for y in self.supported_by for x in y]
            supported_by_str = ", ".join(name_strings)
            string += "\t Supported by:   [" + supported_by_str + "]\n"
        if self.supports_facts != []:
            name_strings = [str(x.name) for x in self.supports_facts]
            supports_f_str = ", ".join(name_strings)
            string += "\t Supports facts: [" + supports_f_str + "]\n"
        if self.supports_rules != []:
            name_strings = [str(x.name) for x in self.supports_rules]
            supports_r_str = ", ".join(name_strings)
            string += "\t Supports rules: [" + supports_r_str + "]\n"
        return string

    def __eq__(self, other):
        # Definisce il comportamento di == quando applicato a questo oggetto
        is_rule = isinstance(other, Rule)
        return is_rule and self.lhs == other.lhs and self.rhs == other.rhs

    def __ne__(self, other):
        # Definisce il comportamento di != quando applicato a questo oggetto
        return not self == other


class Statement(object):
    """Rappresenta una dichiarazione nella knowledge base, ad esempio
        (glucosio paziente1 148), (bmi paziente1 33.6), (inst paziente1 diabetico).
        Le dichiarazioni appaiono nei fatti o nelle regole, sia a sinistra che a destra.
    Attributi:
        terms (listof Term): elenca i termini (Variabili o Costanti) della dichiarazione
        predicate (str): il predicato della dichiarazione, ad esempio inst, glucosio, bmi
    """

    def __init__(self, statement_list=[]):
        """Costruisce una dichiarazione a partire da una lista di elementi.
        Args:
            statement_list (listof str|Term): il primo elemento è il predicato,
                il resto è una sequenza di termini o stringhe da convertire in Term
        """
        super(Statement, self).__init__()
        self.terms = []
        self.predicate = ""

        if statement_list:
            self.predicate = statement_list[0]
            self.terms = [t if isinstance(t, Term) else Term(t) for t in statement_list[1:]]

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Statement({!r}, {!r})'.format(self.predicate, self.terms)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        return "(" + self.predicate + " " + ' '.join((str(t) for t in self.terms)) + ")"

    def __eq__(self, other):
        # Definisce il comportamento di == quando applicato a questo oggetto
        if self.predicate != other.predicate:
            return False
        for self_term, other_term in zip(self.terms, other.terms):
            if self_term != other_term:
                return False
        return True

    def __ne__(self, other):
        # Definisce il comportamento di != quando applicato a questo oggetto
        return not self == other


class Term(object):
    """Rappresenta un termine nella knowledge base, cioè una variabile o una costante.
        Ad esempio '?x' o 'paziente1'.
    Attributi:
        term (Variable|Constant): il valore del termine, una variabile o una costante
    """

    def __init__(self, term):
        """Costruisce un termine convertendo il dato nella forma corretta.
        Args:
            term (Variable|Constant|string): una variabile o costante già istanziata,
                oppure una stringa da convertire nel costruttore giusto
        """
        super(Term, self).__init__()
        from util import is_var
        is_var_or_const = isinstance(term, Variable) or isinstance(term, Constant)
        self.term = term if is_var_or_const else (Variable(term) if is_var(term) else Constant(term))

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Term({!r})'.format(self.term)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        return str(self.term)

    def __eq__(self, other):
        # Definisce il comportamento di == quando applicato a questo oggetto
        return (self is other
                or isinstance(other, Term) and self.term.element == other.term.element
                or ((isinstance(other, Variable) or isinstance(other, Constant))
                    and self.term.element == other.element))

    def __ne__(self, other):
        # Definisce il comportamento di != quando applicato a questo oggetto
        return not self == other


class Variable(object):
    """Rappresenta una variabile usata nelle dichiarazioni.
    Attributi:
        element (str): il nome della variabile, es. '?x'
    """

    def __init__(self, element):
        """Costruisce una variabile.
        Args:
            element (str): il nome della variabile, ad esempio '?x'
        """
        super(Variable, self).__init__()
        self.element = element

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Variable({!r})'.format(self.element)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        return str(self.element)

    def __eq__(self, other):
        # Definisce il comportamento di == quando applicato a questo oggetto
        return (self is other
                or isinstance(other, Term) and self.term.element == other.term.element
                or ((isinstance(other, Variable) or isinstance(other, Constant))
                    and self.term.element == other.element))

    def __ne__(self, other):
        # Definisce il comportamento di != quando applicato a questo oggetto
        return not self == other


class Constant(object):
    """Rappresenta una costante usata nelle dichiarazioni.
    Attributi:
        element (str): il valore della costante, ad esempio 'diabetico' o 'paziente1'
    """

    def __init__(self, element):
        """Costruisce una costante.
        Args:
            element (str): il valore della costante, ad esempio 'diabetico' o 'paziente1'
        """
        super(Constant, self).__init__()
        self.element = element

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Constant({!r})'.format(self.element)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        return str(self.element)

    def __eq__(self, other):
        # Definisce il comportamento di == quando applicato a questo oggetto
        return (self is other
                or isinstance(other, Term) and self.term.element == other.term.element
                or ((isinstance(other, Variable) or isinstance(other, Constant))
                    and self.term.element == other.element))

    def __ne__(self, other):
        # Definisce il comportamento di != quando applicato a questo oggetto
        return not self == other


class Binding(object):
    """Rappresenta il legame tra una variabile e una costante, ad esempio
        '?x' può essere associata a 'paziente1'.
    Attributi:
        variable (Variable): la variabile coinvolta nel binding
        constant (Constant): la costante assegnata alla variabile
    """

    def __init__(self, variable, constant):
        """Costruisce un binding tra variabile e costante.
        Args:
            variable (Variable): la variabile associata a questo binding
            constant (Constant): il valore associato alla variabile
        """
        super(Binding, self).__init__()
        self.variable = variable
        self.constant = constant

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Binding({!r}, {!r})'.format(self.variable, self.constant)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        return self.variable.element.upper() + " : " + self.constant.element


class Bindings(object):
    """Rappresenta i binding usati quando si confrontano due dichiarazioni.
    Attributi:
        bindings (listof Bindings): lista dei binding creati durante il match
        bindings_dict (dictof Bindings): dizionario dei binding dove la chiave è
            la variabile e il valore è la costante associata,
            ad esempio bindings_dict['?x'] => 'paziente1'
    """

    def __init__(self):
        # Costruttore per Bindings: crea un'istanza inizialmente vuota
        self.bindings = []
        self.bindings_dict = {}

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'Bindings({!r}, {!r})'.format(self.bindings_dict, self.bindings)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        if self.bindings == []:
            return "No bindings"
        return ", ".join((str(binding) for binding in self.bindings))

    def __getitem__(self, key):
        """Definisce l'indicizzazione: restituisce bindings_dict[key] se presente, altrimenti None."""
        return (self.bindings_dict[key]
                if (self.bindings_dict and key in self.bindings_dict)
                else None)

    def add_binding(self, variable, value):
        """Aggiunge un binding tra variabile e valore.
        Args:
            variable (Variable): la variabile da associare
            value (Constant): la costante da associare alla variabile
        """
        self.bindings_dict[variable.element] = value.element
        self.bindings.append(Binding(variable, value))

    def bound_to(self, variable):
        """Verifica se una variabile ha un binding già assegnato.
        Args:
            variable (Variable): variabile da controllare
        Returns:
            Variable|Constant|False: ritorna il termine vincolato se esiste, altrimenti False
        """
        if variable.element in self.bindings_dict.keys():
            value = self.bindings_dict[variable.element]
            if value:
                from util import is_var
                return Variable(value) if is_var(value) else Constant(value)

        return False

    def test_and_bind(self, variable_term, value_term):
        """Verifica se una variabile è già vincolata. Se sì controlla se il valore coincide.
            Altrimenti aggiunge il binding e restituisce True.
        Args:
            value_term (Term): valore da assegnare
            variable_term (Term): variabile da vincolare
        Returns:
            bool: se la variabile è vincolata restituisce se il valore corrisponde, altrimenti True
        """
        bound = self.bound_to(variable_term.term)
        if bound:
            return value_term.term == bound

        self.add_binding(variable_term.term, value_term.term)
        return True


class ListOfBindings(object):
    """Contenitore per più binding.
        Attributi:
            list_of_bindings (listof Bindings): raccoglie i binding trovati
    """

    def __init__(self):
        # Costruttore per ListOfBindings
        super(ListOfBindings, self).__init__()
        self.list_of_bindings = []

    def __repr__(self):
        # Rappresentazione interna in formato stringa
        return 'ListOfBindings({!r})'.format(self.list_of_bindings)

    def __str__(self):
        # Rappresentazione esterna quando stampata
        string = ""
        for binding, associated_fact_rules in self.list_of_bindings:
            string += "Bindings for Facts and Rules: " + str(binding) + "\n"
            string += "Associated Facts and Rules: ["
            string += ", ".join((str(f) for f in associated_fact_rules)) + "]\n"
        return string

    def __len__(self):
        """Definisce il comportamento di len su questa classe,
            ad esempio len(ListOfBindings([])) == 0
        """
        return len(self.list_of_bindings)

    def __getitem__(self, key):
        """Definisce l'indicizzazione per questa lista di binding.
            Ad esempio list_of_bindings[i] restituisce il primo elemento della coppia.
        """
        return self.list_of_bindings[key][0]

    def add_bindings(self, bindings, facts_rules=[]):
        """Aggiunge binding alla lista insieme ai fatti o alle regole associate.
            Args:
                bindings (Bindings): binding da aggiungere
                facts_rules (listof Fact|Rule): regole o fatti associati al binding
            Returns:
                None
        """
        self.list_of_bindings.append((bindings, facts_rules))


class ClassificationResult:
    """Contenitore delle metriche usate per valutare ogni modello di classificazione.
        Attributi:
            model(str): nome del modello di classificazione
            accuracy(float): accuratezza ottenuta dal modello
            f1(float): F1-score ottenuto dal modello
            precision(float): precisione ottenuta dal modello
            recall(float): richiamo ottenuto dal modello
            balanced_accuracy(float): balanced accuracy ottenuta dal modello
            auc(float): valore AUC ottenuto dal modello
            kmeans_score(str): valori del punteggio K-Means se applicabile
            calc_class(str): percorso del dataset utilizzato per la classificazione
    """
    model = ''
    accuracy = float
    f1 = float
    precision = float
    recall = float
    balanced_accuracy = float
    auc = float
    kmeans_score = {}
    calc_class = ''

    def __init__(self, model, accuracy, f1, precision, recall, balanced_accuracy, auc, kmeans_score, calc_class):
        # Costruttore per ClassificationResult
        self.model = model
        self.accuracy = accuracy
        self.f1 = f1
        self.precision = precision
        self.recall = recall
        self.balanced_accuracy = balanced_accuracy
        self.auc = auc
        self.kmeans_score = kmeans_score
        self.calc_class = calc_class
