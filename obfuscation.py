import ast
import codegen
import astor
import copy
import numpy as np

def swap_if_branches(node):
    assert type(node) == ast.If, type(node)
    
    if node.orelse:
#         print("swapping")
        #print(ast.dump(node))
        true_path = node.body
        false_path = node.orelse
        
        old_condition = node.test
        new_condition = ast.UnaryOp(op=ast.Not(), operand=old_condition)
        
        node.test = new_condition
        node.body = false_path
        node.orelse = true_path
        
    
    return node

def get_argument_names(args):
    names = set()
    for arg in args.args:
        names.add(arg.arg)
        
    if args.vararg is not None:
        names.add(args.vararg.arg)
    
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
        
    return names

def generate_name(names):
    name = "dummy"
    id = 0
    while name + "_" + str(id) in names:
        id += 1

    return name + "_" + str(id)

def add_args(node, n, pad_default):
    assert type(node) == ast.FunctionDef
    args = node.args
    names = get_argument_names(args)
    
    for i in range(n):
        new_name = generate_name(names)
        names.add(new_name)
        new_arg = ast.arg(arg=new_name, annotation=None)
        args.args.append(new_arg)
        if pad_default:
            args.defaults.append(ast.Num(0))
        
    return node


def add_args_kvargs(node, varargs, kvargs):
    assert type(node) == ast.FunctionDef
    args = node.args
    names = get_argument_names(args)
    if varargs:
        if args.vararg is None:
            new_name = generate_name(names)
            names.add(new_name)
            args.vararg = ast.arg(arg=new_name, annotation=None)
            
    
    if varargs:
        if args.kwarg is None:
            new_name = generate_name(names)
            names.add(new_name)
            args.kwarg = ast.arg(arg=new_name, annotation=None)
    
    return node

def get_keyword_call_names(node):
    assert type(node) == ast.Call, type(node)
    
    names = set()
    for kw in node.keywords:
        names.add(kw.arg)
    
    
    return names
    
    
    
def add_arguments_to_call(node, n_positional, n_kwargs):
    assert type(node) == ast.Call, type(node)
    
    args = node.args
    keywords = node.keywords
    
    
    
    for i in range(n_positional):
        args.append(ast.Name(id="dummy", ctx=ast.Load()))
        
    names = get_keyword_call_names(node)
    for i in range(n_kwargs):
        new_name = generate_name(names)
        names.add(new_name)
        keywords.append(ast.keyword(arg=new_name, value=ast.Num(0)))
    
    
    return node


def swap_elements_in_body(node, pos_1, pos_2):
    assert hasattr(node, "body")
    
    body = node.body
    body[pos_1], body[pos_2] = body[pos_2], body[pos_1]
    
    return node

trash_code = """
import no_import

def trash():
    pass

@trash
def empty_decorator():
    pass

pass

while False:
    pass
    
if False:
    pass
"""

parsed_trash = ast.parse(trash_code)

def traverse_extract(node):
    result = []
    
    if type(node) == ast.FunctionDef:
        result.append(copy.deepcopy(node))
        
    if hasattr(node, "body") and isinstance(node.body, list):
        result.append(copy.deepcopy(node.body))
    
    for child in ast.iter_child_nodes(node):
        result += traverse_extract(child)
    
    return result

def extract_patches(node):
    node = copy.deepcopy(node)
    
    return traverse_extract(node)
    

def add_patches_to_code(node, patches, n, params):
    node = copy.deepcopy(node)
    assert hasattr(node, "body") and isinstance(node.body, list)
    
    for i in range(n):
        position = np.random.randint(max(len(node.body), 1))
        
#         print(">>>>>", patches)
        to_insert = patches[np.random.randint(len(patches))]
        if isinstance(to_insert, list):
            to_insert = to_insert[:np.random.randint(len(to_insert))]
            if len(to_insert) > params['max_frac_to_insert'] * len(node.body):
                continue
            node.body = node.body[:position] + copy.deepcopy(to_insert) + node.body[position:]
        elif params['add_functions']:
#             continue
            node.body = node.body[:position] + [copy.deepcopy(to_insert)] + node.body[position:]
    
    
    return node

def traverse_add_patches(node, patches, params):
    node = copy.deepcopy(node)
    
    for child in ast.iter_child_nodes(node):
        traverse_add_patches(child, patches, params)
        
        
    if hasattr(node, "body") and isinstance(node.body, list) and np.random.rand() < params['modify_add_patches']:
        node = add_patches_to_code(node, patches, np.random.randint(params['max_patches_to_body']), params)
    
    return node
        
        

def modify_insert_code(a, b, params):
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)

    patches = extract_patches(b)
#     print(patches)
    
    result = traverse_add_patches(a, patches, params)
    
    return result
    
    
def add_trash_to_body(node, n):
    assert hasattr(node, "body")
    
    for i in range(n):
        position = np.random.randint(max(len(node.body), 1))
        
        to_insert = np.random.choice(parsed_trash.body)
        node.body = node.body[:position] + [copy.deepcopy(to_insert)] + node.body[position:]
    
    
    return node


def modify(node, params):
    node = copy.deepcopy(node)
    
    if type(node) == ast.FunctionDef:
        if np.random.rand() < params['add_kvargs']:
            add_args_kvargs(node, bool(np.random.randint(2)), bool(np.random.randint(2)))
        
        if np.random.rand() < params['add_args']:
            add_args(node, np.random.randint(params['args_max_add']), pad_default=np.random.randint(2))
        
    if type(node) == ast.Call and np.random.rand() < params['add_call_args']:
        add_arguments_to_call(node, np.random.randint(params['call_args']), np.random.randint(params['call_kwargs']))
        
    if type(node) == ast.If and np.random.rand() < params['if_swap']:
        swap_if_branches(node)
        
    for child in ast.iter_child_nodes(node):
        modify(child, params)
        
    if hasattr(node, "body") and isinstance(node.body, list) and np.random.rand() < params['modify_body']:
#         node.body += ast.parse(code).body
#         swap_elements_in_body(node, 0, 1)
        node = add_trash_to_body(node, np.random.randint(params['max_trash_to_body']))
        if node.body  and np.random.rand() < params['swap_in_body']:
            swap_elements_in_body(node, np.random.randint(len(node.body)), np.random.randint(len(node.body)))
    
    
    
    
    return node


example_params = {
    'add_kvargs':0.4,
    'add_args':0.5,
    'args_max_add':3,
    'add_call_args':0.6,
    'call_args':3,
    'call_kwargs':3,
    'if_swap':0.5,
    'modify_body':0.4,
    'n_modification_depth':3,
    'max_trash_to_body':5,
    'swap_in_body':0.4
}

def obfuscate_add_trash(code, params):
    parsed = ast.parse(code)
    
    result = parsed
    for i in range(params['n_modification_depth']):
        result = modify(result, params)
    
    
    return astor.code_gen.to_source(result)

def obfuscate_extract_fragments(code, params):
    return code


def obfuscate(code, params):
    t = params.get("type", "add_trash")
    if t == "add_trash":
        return obfuscate_add_trash(code, params)
    elif t == "extract_fragments":
        return obfuscate_extract_fragments(code, params)
    else:
        raise ValueError("Incorrect obfuscation type " + t)
    


def obfuscate_mixed(code, other, params):

    result = modify_insert_code(ast.parse(code), ast.parse(other), params)
    
    
    return astor.code_gen.to_source(result)
    
        