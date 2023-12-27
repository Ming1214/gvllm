import guidance


SPACE = guidance.zero_or_more(" ")
LINEFEED = guidance.zero_or_more("\n")
SPACE_OR_LINEFEED = guidance.zero_or_more(guidance.select([" ", "\n"]))


def convert_json_case_to_grammar(case):
    if isinstance(case, str):
        return '"' + guidance.gen(stop = '"')
    if isinstance(case, int):
        return guidance.gen(regex = "-?\d+")
    if isinstance(case, float):
        return guidance.gen(regex = "-?\d+(\.\d*)?")
    if isinstance(case, bool):
        return guidance.select(["true", "false"])
    if isinstance(case, list):
        grammar = convert_json_case_to_grammar(case[0])
        return "[" + SPACE_OR_LINEFEED + grammar + guidance.zero_or_more("," + SPACE_OR_LINEFEED + grammar) + SPACE_OR_LINEFEED + "]"
    if isinstance(case, dict):
        grammar = '{' + SPACE_OR_LINEFEED
        for i, (key, value) in enumerate(case.items()):
            grammar += f'"{key}"' + SPACE_OR_LINEFEED + ':' + SPACE_OR_LINEFEED
            grammar += convert_json_case_to_grammar(value)
            if i < len(case)-1:
                grammar += ',' + SPACE_OR_LINEFEED
        grammar += SPACE_OR_LINEFEED + '}'
        return grammar
    raise TypeError(f"Unsupported Type: {type(case)}")


def convert_json_schema_to_grammar(schema):
    raise NotImplementedError