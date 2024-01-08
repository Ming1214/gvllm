import guidance
from enum import Enum


class GuidanceSpliter(Enum):
    
    NONE: 0 = ""
    SPACE: 1 = " "
    LINEFEED: 2 = "\n"
    
    NONE_OR_SPACE: 3 = guidance.select(["", " "])
    NONE_OR_LINEFEED: 4 = guidance.select(["", "\n"])
    NONE_OR_SPACE_OR_LINEFEED: 5 = guidance.select(["", " ", "\n"])
    
    ZERO_OR_MORE_SPACE: 6 = guidance.zero_or_more(" ")
    ZERO_OR_MORE_LINEFEED: 7 = guidance.zero_or_more("\n")
    ZERO_OR_MORE_SPACE_OR_LINEFEED: 8 = guidance.zero_or_more(guidance.select([" ", "\n"]))
    
    ONE_OR_MORE_SPACE: 9 = guidance.one_or_more(" ")
    ONE_OR_MORE_LINEFEED: 10 = guidance.one_or_more("\n")
    ONE_OR_MORE_SPACE_OR_LINEFEED: 11 = guidance.one_or_more(guidance.select([" ", "\n"]))


def convert_json_case_to_grammar(case, SPLITER = ""):
    if isinstance(case, str):
        return '"' + guidance.gen(stop = '"')
    if isinstance(case, int):
        return guidance.gen(regex = "[\-\+]?\d+")
    if isinstance(case, float):
        return guidance.gen(regex = "[\-\+]?\d+(\.\d*)?")
    if isinstance(case, bool):
        return guidance.select(["true", "false"])
    if isinstance(case, list):
        grammar = convert_json_case_to_grammar(case[0], SPLITER)
        return "[" + SPLITER + grammar + guidance.zero_or_more("," + SPLITER + grammar) + SPLITER + "]"
    if isinstance(case, dict):
        grammar = '{' + SPLITER
        for i, (key, value) in enumerate(case.items()):
            grammar += f'"{key}":' + SPLITER
            grammar += convert_json_case_to_grammar(value, SPLITER)
            if i < len(case)-1:
                grammar += ',' + SPLITER
        grammar += SPLITER + '}'
        return grammar
    raise TypeError(f"Unsupported Type: {type(case)}")


def convert_json_schema_to_grammar(schema, definitions = None, SPLITER = "", temperature = -1):
    
    if definitions is None:
        definitions = schema.get("definitions", None)
    ref = schema.get("$ref", None)
    if ref is not None:
        ref = ref.split("/")[-1]
        assert definitions is not None and ref in definitions, Exception(f"Can't find $ref: {ref} from {definition}!")
        return convert_json_schema_to_grammar(definitions[ref], definitions, SPLITER, temperature = temperature)
    
    if "enum" in schema:
        type = "enumeration"
        type_enum = schema.get("type", None)
        if type_enum is None and isinstance(schema["enum"][0], str):
            type_enum = "string"
    else:
        type = schema.get("type", None)
    if type is None:
        raise TypeError(f"Unrecognized type: {schema}")

    temperature = schema.get("temperature", temperature)
    
    if type == "boolean":
        return guidance.select(["true", "false"])
        
    elif type == "integer":
        return guidance.gen(regex = "[\-\+]?\d+", temperature = temperature)
        
    elif type == "number":
        return guidance.gen(regex = "[\-\+]?\d+(\.\d*)?", temperature = temperature)
        
    elif type == "string": # 支持正则表达式(constr)
        if "pattern" in schema:
            return '"' + guidance.gen(regex = schema["pattern"], stop = '"', temperature = temperature)
        return '"' + guidance.gen(stop = '"', temperature = temperature)
        
    elif type == "enumeration":
        if type_enum == "string":
            return '"' + guidance.select([x.strip("'\"")]) + '"'
        return guidance.select(schema["enum"])
        
    elif type == "array": # 支持设定数目限制(conlist)
        min_items = max(schema.get("min_items", 0), 0)
        max_items = schema.get("max_items", -1)
        if min_items == max_items == 0: return "[]" # 一般不存在这种情况
        item_grammar = convert_json_schema_to_grammar(schema["items"], definitions, SPLITER, temperature = temperature)
        grammar = "["
        for i in range(min_items):
            grammar += SPLITER + item_grammar
            if i < min_items-1:
                grammar += ","
        if max_items < min_items: # 无上限
            continue_grammar = guidance.zero_or_more("," + SPLITER + item_grammar) + SPLITER + "]"
            if min_items > 0:
                grammar += continue_grammar
            else:
                grammar += guidance.select(["]", SPLITER + item_grammar + continue_grammar])
        else: # 有上限
            if min_items > 0:
                for i in range(max_items-min_items):
                    grammar += guidance.select(["", "," + SPLITER + item_grammar])
                grammar += SPLITER + "]"
            else:
                rest_grammar = SPLITER + item_grammar
                for i in range(max_items-1):
                    rest_grammar += guidance.select(["", "," + SPLITER + item_grammar])
                grammar += guidance.select(["]", rest_grammar + SPLITER + "]"])
        return grammar
        
    elif type == "object":
        properties = schema.get("properties", {})
        if len(properties) == 0: return "{}"
        #required = schema.get("required", [])
        grammar = "{" + SPLITER
        for i, key in enumerate(properties):
            value = properties[key]
            value_grammar = convert_json_schema_to_grammar(value, definitions, SPLITER, temperature = temperature)
            grammar += f'"{key}":' + SPLITER + value_grammar
            if i < len(properties)-1:
                grammar += "," + SPLITER
        grammar += SPLITER + "}"
        return grammar
        
    else: raise TypeError(f"Unsupported Type: {type}")
