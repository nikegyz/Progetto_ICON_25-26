from logical_classes import *


# Legge e tokenizza un file di fatti e regole.
def read_tokenize(file):
    """Legge un file e lo trasforma in liste di fatti e regole.
    Args:
        file (file): un file di testo con fatti del tipo
            (predicato soggetto oggetto), ad esempio
            "fact: (isa diabetico paziente)".
            Sono anche previste regole con parte sinistra e destra, ad esempio
            "rule: ((inst ?x ?y) (isa ?y ?z)) -> (inst ?x ?z)".
            Ogni fatto o regola deve essere su una riga separata.
    Returns:
        list: una lista di oggetti Fact e Rule.
    """
    file = open(file, "r")
    elements = []
    current = ""
    for line in file:
        line_stripped = line.strip()
        if not line_stripped or line_stripped.startswith('#'):
            continue
        if line_stripped.startswith("fact:") or line_stripped.startswith("rule:"):
            elements.append(current)
            current = line_stripped
        else:
            current = current + " " + line_stripped
    elements.append(current)
    output = []
    for e in elements:
        parsed = parse_input(e)
        if isinstance(parsed, Fact) or isinstance(parsed, Rule):
            output.append(parsed)
    file.close()
    return output


def parse_input(e):
    """Analizza una stringa di input, etichettando i fatti e separando le regole.
    Args:
        e (str): stringa da analizzare
    Returns:
        Fact|Rule|str|None: oggetto Fact o Rule per input validi, la stringa per commenti,
            None per input vuoti.
    """
    if len(e) == 0:
        # return (BLANK, None)
        return None
    elif e[0] == '#':
        # return (COMMENT, e)
        return e[1:]
    elif e[0:5] == "fact:":
        e = e[5:].replace(")", "").replace("(", "").rstrip().strip().split()
        # return (FACT, e)
        return Fact(e)
    elif e[0:5] == "rule:":
        e = e[5:].split("->")
        rhs = e[1].replace(")", "").replace("(", "").rstrip().strip().split()
        lhs = e[0].rstrip(") ").strip("( ").replace("(", "").split(")")
        lhs = map(lambda x: x.rstrip().strip().split(), lhs)
        # return (RULE, [lhs, rhs])
        return Rule([lhs, rhs])
    else:
        print("PARSE ERROR: input header", e[0:5], "not recognized.")


def get_new_fact_or_rule():
    """Crea un nuovo fatto o una nuova regola chiedendolo all'utente.
    Returns:
        Fact|Rule: il fatto o la regola inserita dall'utente
    """
    msg = "Digita un nuovo fatto o una nuova regola da aggiungere alla KB:\n"
    e = input(msg)
    return parse_input(e)


def get_new_statements():
    """Richiede all'utente una dichiarazione in forma predicato + termini.
    Le dichiarazioni devono essere del tipo "pred x1 x2 ...".
    Questa funzione supporta anche nomi che contengono il carattere "?",
    utile per variabili come ?x o ?y.
    Returns:
        list: lista di token della dichiarazione inserita dall'utente
    """
    e = input("Digita una dichiarazione del tipo pred x1 x2 ... :\n")
    return e.split()
