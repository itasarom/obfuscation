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
    
    
def add_trash_to_body(node, n):
    assert hasattr(node, "body")
    
    for i in range(n):
        position = np.random.randint(max(len(node.body), 1))
        
        to_insert = np.random.choice(parsed_trash.body)
        node.body = node.body[:position] + [copy.deepcopy(to_insert)] + node.body[position:]
    
    
    return node


def modify(node, params):
    
    
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
def obfuscate(code, params):
    parsed = ast.parse(code)
    
    result = parsed
    for i in range(params['n_modification_depth']):
        result = modify(result, params)
    
    
    return astor.code_gen.to_source(result)
    
        