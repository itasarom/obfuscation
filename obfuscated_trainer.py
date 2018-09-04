
from collections import defaultdict
import numpy as np
import torch
import ast
import matplotlib.pyplot as plt
import obfuscation
import astor
from IPython import display
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import os
import pickle as pkl
import copy

def sample_from_data(data):
    n_users = len(data)
    current_user = list(sorted(data.keys()))[np.random.choice(n_users)]
    current_data = data[current_user]
    cur_len = len(current_data)
    code_id = list(sorted(current_data.keys()))[np.random.choice(cur_len)]
    code = current_data[code_id]
    
    return code

def sample_from_data_adversarial(data, n_adversarial):
    n_users = len(data)
    current_user = list(sorted(data.keys()))[np.random.choice(n_users)]
#     print("Current user:", current_user)
    current_data = data[current_user]
    cur_len = len(current_data)
    code_id = list(sorted(current_data.keys()))[np.random.choice(cur_len)]
    code = current_data[code_id]
    
    user_set = list(set(data.keys()) - {current_user})
    user_list = list(sorted(user_set))
    adversarial_list = []
    for i in range(n_adversarial):
        current_user = user_list[np.random.choice(n_users - 1)]
#         print("\tAdversarial user: ", current_user)
        current_data = data[current_user]
        cur_len = len(current_data)
        code_id = list(sorted(current_data.keys()))[np.random.choice(cur_len)]
        current_code = current_data[code_id]
        adversarial_list.append(current_code)
    
    return code, adversarial_list


class Batcher:
    def __init__(self, data, train_size, seed=42):
        self.train_data = defaultdict(dict)
        self.test_data = defaultdict(dict)
        self.classes = list(sorted(data.keys()))
        
        for handle in sorted(data.keys()):
            codes_for_handle = data[handle]
            n_train = int(len(codes_for_handle) * train_size)
            #is it correct to use sorted problem names?
            current_train = {}
            current_test = {}
            for id, problem in enumerate(sorted(codes_for_handle.keys())):
                current_code = codes_for_handle[problem]
                if id < n_train:
                    current_train[problem] = current_code
                else:
                    current_test[problem] = current_code
            
            self.train_data[handle] = current_train
            self.test_data[handle] = current_test
            
    def dump(self, dir_name, override=False):
        os.makedirs(dir_name, exist_ok=override)
        with open(os.path.join(dir_name, "batcher_state.pkl"), "wb") as f:
            d = {
                "train_data":self.train_data,
                "test_data":self.test_data,
                "classes":self.classes
            }

            pkl.dump(d, f)
            
    def load(self, dir_name):
        with open(os.path.join(dir_name, "batcher_state.pkl"), "rb") as f:
            d = pkl.load(f)
            self.train_data = d['train_data']
            self.test_data = d['test_data']
            self.classes = d['classes']
            
    
    def load_from_attribution_batcher(self, dir_name):
        with open(os.path.join(dir_name, "batcher_state.pkl"), "rb") as f:
            d = pkl.load(f)
            
            self.classes = d['classes']
            
            classes_back = {id:name for id, name in enumerate(self.classes)}
            if hasattr(self, "raw_x_train"):
                train_x = d["raw_x_train"]
                test_x = d["raw_x_test"]
            else:
                train_x = d["x_train"]
                test_x = d["x_test"]
            
            def get_data(x, y):
                assert len(x) == len(y)
                result = defaultdict()
                for x_cur, y_cur in zip(x, y):
                    name_cur = classes_back[y_cur]
                    if name_cur not in result:
                        result[name_cur] = defaultdict(list)
                        
                    cur = result[name_cur]
                    cur[len(cur)] = x_cur
                
                return result
                    
             
            self.train_data = get_data(train_x, d["y_train"])
            self.test_data = get_data(test_x, d["y_test"])
        
                
            
    def train_code(self, n_problems, n_adversarial):
        for i in range(n_problems):
            yield sample_from_data_adversarial(self.train_data, n_adversarial)
            
    
    def validation_code(self, n_problems, n_adversarial):
        for i in range(n_problems):
            yield sample_from_data_adversarial(self.train_data, n_adversarial)
            

def loss_for_problem(ast_encoder, code, n_tests, obfuscation_params):
#     sns.heatmap(get_multi_similarity_matrix(ast_encoder, code, n_tests, obfuscation_params)[0])
#     plt.show()
    obfuscated_repr, obfuscated = get_n_obfuscated_representations(ast_encoder, code, n_tests, obfuscation_params)
    obfuscated_repr = torch.cat(obfuscated_repr, dim=0)
    
    original_repr = ast_encoder(ast.parse(code)).view(1, -1)
    
    original_repr = original_repr.repeat(n_tests, 1)
    
#     print(cosine_similarity(original_repr.detach(), obfuscated_repr.detach()))
#     print(torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr))
    loss = torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr).mean()
    
    
    return loss, obfuscated_repr, obfuscated

def loss_adversarial(ast_encoder, code, code_list, obfuscation_params):
    
    
    
    all_repr = []
    all_obfuscated_code = []
    for adversarial_code in code_list:
        obfuscated_repr, obfuscated = get_n_obfuscated_representations(ast_encoder, adversarial_code, 1, obfuscation_params)
#         obfuscated_repr = [ast_encoder(ast.parse(adversarial_code)).view(1, -1)]
#         obfuscated = []

        all_repr += obfuscated_repr
        all_obfuscated_code += obfuscated
    
    
    
    original_repr = ast_encoder(ast.parse(code)).view(1, -1)
    
    if not code_list:
        return 0.0, torch.zeros_like(original_repr), []
    
    original_repr = original_repr.repeat(len(all_repr), 1)

    
                                             
    obfuscated_repr = torch.cat(all_repr, dim=0)
    
#     print(torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr))
    loss = torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr).mean()
    
    
    return loss, obfuscated_repr, obfuscated
 
    
def get_n_obfuscated_representations(ast_encoder, code, n_tests, obfuscation_params):
    data_points = []
    obfuscated_list = []
    for i in range(n_tests):
#             print(code)
            obfuscated = obfuscation.obfuscate(code, obfuscation_params)
#             print(obfuscated)
            obfuscated_list.append(obfuscated)
            obfuscated_repr = ast_encoder(ast.parse(obfuscated)).view(1, -1)
            data_points.append(obfuscated_repr)
#     print("OK")
    return data_points, obfuscated_list

def get_self_similarity_matrix(ast_encoder, code, n_tests, obfuscation_params):
    data_points = [ast_encoder(ast.parse(code)).view(1, -1)]
    data_points += get_n_obfuscated_representations(ast_encoder, code, n_tests, obfuscation_params)[0]
        
    data_points = torch.cat(data_points, dim=0)
    
    data_points.detach().numpy()
    
    return cosine_similarity(data_points, data_points)

def get_multi_similarity_matrix(ast_encoder, code_list, n_tests, obfuscation_params):
    data_points = []
    indices_by_user = []
    for code in code_list:
        indices_by_user.append(list(range(len(data_points), len(data_points) + n_tests + 1, 1)))
        data_points.append(ast_encoder(ast.parse(code)).view(1, -1))
        data_points += get_n_obfuscated_representations(ast_encoder, code, n_tests, obfuscation_params)[0]
        
    
    data_points = torch.cat(data_points, dim=0).detach().numpy()
    return cosine_similarity(data_points, data_points), indices_by_user

# The old version
# def precision_higher(sim, indices, data_indices):
#     indices_by_user = []
#     for data_index in data_indices:
#         current = []
#         for id in data_index:
#             current += indices[id]
#         indices_by_user.append(current)
    
#     result = []
#     n_cols = sim.shape[1]
#     for cls_id, indices in enumerate(indices_by_user):
        
#         current_correct = 0
#         for row in indices:
#             argmax = sim[row, list(range(0, row, 1)) + list(range(row + 1, n_cols, 1))].argmax()
#             if argmax in indices:
#                 current_correct += 1
        
#         result.append(current_correct/len(indices))
     
#     return result

def precision_higher(sim, indices, data_indices):
    indices_by_user = []
    for data_index in data_indices:
        current = []
        for id in data_index:
            current += indices[id]
        indices_by_user.append(current)
    
    result = []
    n_cols = sim.shape[1]
    for cls_id, indices in enumerate(indices_by_user):
        
        current_correct = 0
        for row in indices:
            argmaxes = (-sim[row]).argsort()
            for argmax in argmaxes:
                if argmax != row:
                    break
            if argmax in indices:
                current_correct += 1
        
        result.append(current_correct/len(indices))
     
    return result

def precision_lower(sim, indices, data_indices):
    indices_by_user = []
    for data_index in data_indices:
        current = []
        for id in data_index:
            current.append(indices[id])
        indices_by_user.append(current)
    
    result = []
    result_for_original = []
    n_cols = sim.shape[1]
    total_problems = 0
    for cls_id, indices in enumerate(indices_by_user):
       
        current_correct = 0
        current_correct_original = 0
        current_length = 0
        for current_problem in indices:
#             print(current_problem)
#             print(indices)
            for id, row in enumerate(current_problem):
                
#                 row += total_problems
                argmaxes = (-sim[row]).argsort()
                for argmax in argmaxes:
                    if argmax != row:
                        break
                        
                if argmax in current_problem:
                    current_correct += 1
                
                    if id == 0:
                        current_correct_original += 1
                    
            
            total_problems += len(current_problem)
            current_length += len(current_problem)
        
        result.append(current_correct/current_length)
        result_for_original.append(current_correct_original/len(indices))
     
    return result, result_for_original

def create_similarity_matrix(ast_encoder, names, data, obfuscation_params, n_first_for_person, n_obfuscated):
    ast_encoder.eval()
    data_points = []
    data_indices = []
    for name in names:
        start = len(data_points)
        data_points += list(data[name].values())[:n_first_for_person]
        end = len(data_points)
        data_indices.append(list(range(start, end, 1)))

    sim, indices = get_multi_similarity_matrix(ast_encoder, data_points, n_obfuscated, obfuscation_params)
#     plt.figure(figsize=(15, 15))
#     sns.heatmap(sim, vmin=-1.0, vmax=1.0)
    
    return sim, indices, data_indices


def order_names_by_count(batcher):
    lengths = []
    for name in batcher.classes:
        lengths.append(len(batcher.train_data[name]))
    
#     print(lengths)
    
    return np.array(batcher.classes)[np.argsort(-np.array(lengths))]

def print_validation_result(sim, indices, data_indices):
    acc = precision_lower(sim, indices, data_indices)
    print(np.mean(acc[1]))
    print(np.mean(acc[0]))
    # acc[0]
    acc = precision_higher(sim, indices, data_indices)
    print(np.mean(acc))
    plt.figure(figsize=(15, 15))
    sns.heatmap(sim, vmin=-1.0, vmax=1.0)
    plt.show()

def validate(ast_encoder, batcher, long_names, obfuscation_params, n_first_for_person, n_obfuscated):
    sim, indices, data_indices = create_similarity_matrix(ast_encoder, long_names, batcher.train_data, obfuscation_params, n_first_for_person=n_first_for_person, n_obfuscated=n_obfuscated)
    print("Train:")
    print_validation_result(sim, indices, data_indices)
    
    sim, indices, data_indices = create_similarity_matrix(ast_encoder, long_names, batcher.test_data, obfuscation_params, n_first_for_person=n_first_for_person, n_obfuscated=n_obfuscated)
    print("Test:")
    print_validation_result(sim, indices, data_indices)


def dump(trainer, dir_name, override=False):
    os.makedirs(dir_name, exist_ok=override)
    cls = trainer.ast_encoder
    torch.save(cls.state_dict(), os.path.join(dir_name, "model_state.tc"))
    torch.save(trainer.optimizer, os.path.join(dir_name, "optimizer.tc"))
    torch.save((trainer.train_metrics, trainer.validation_metrics), os.path.join(dir_name, "metrics.tc"))
#     torch.save((trainer.current_epoch, trainer.all_params), os.path.join(dir_name, "params_epochs.tc"))
    
    cls_path = os.path.join(dir_name, "components")
    os.makedirs(cls_path, exist_ok=override)
    cls.save(cls_path)
    

class Trainer:
    def __init__(self, ast_encoder, optimizer, track_metric, path):
        self.ast_encoder = ast_encoder
        self.optimizer = optimizer
        
        self.path = path
        self.track_metric = track_metric
#         optimizer = torch.optim.Adam(ast_encoder.parameters())
#         train_loss = []
#         validation_loss = []
#         validation_iterations = []
        self.train_metrics = defaultdict(list)
        self.validation_metrics = defaultdict(list)
        self.all_params = []
        
    def plot_all(self):
            display.clear_output()
            plt.figure(figsize=(15, 10))
            plt.plot(self.train_metrics['loss'], label='train')
            plt.plot(self.validation_metrics['iterations'], self.validation_metrics['loss'], label='test')
            plt.xlabel("epochs")
            plt.ylabel("cosine similarity")
            plt.legend()
            plt.show()
            
            plt.figure(figsize=(15, 10))
            plt.plot(self.train_metrics['regularizer'], label='train')
            plt.plot(self.validation_metrics['iterations'], self.validation_metrics['regularizer'], label='val')
            plt.legend()
            plt.show()
            
            plt.figure(figsize=(15, 10))
            plt.plot(self.train_metrics['adversarial'], label='train')
            plt.plot(self.validation_metrics['iterations'], self.validation_metrics['adversarial'], label='val')
            plt.legend()
            plt.show()
            
    def is_best_state(self):
        metric_name = self.track_metric['metric_name']
        if self.track_metric['function'](self.validation_metrics[metric_name]):
            self.dump(os.path.join(self.path, "best_state"), override=True)
            
    def dump(self, dir_name, override=False):
        os.makedirs(dir_name, exist_ok=override)
        ast_encoder = self.ast_encoder
        torch.save(ast_encoder.state_dict(), os.path.join(dir_name, "model_state.tc"))
        torch.save(self.optimizer, os.path.join(dir_name, "optimizer.tc"))
        torch.save((self.train_metrics, self.validation_metrics), os.path.join(dir_name, "metrics.tc"))
        torch.save(self.all_params, os.path.join(dir_name, "params_epochs.tc"))

        cls_path = os.path.join(dir_name, "components")
        os.makedirs(cls_path, exist_ok=override)
        ast_encoder.save(cls_path)
    
    def load(self, dir_name):
        self.ast_encoder.load_state_dict(torch.load(os.path.join(dir_name, "model_state.tc")))
        self.optimizer = torch.load(os.path.join(dir_name, "optimizer.tc"))
        (self.train_metrics, self.validation_metrics) = torch.load(os.path.join(dir_name, "metrics.tc"))
        self.all_params = torch.load(os.path.join(dir_name, "params_epochs.tc"))
        
        
    def train(self, batcher, params, obfuscation_params):
        self.all_params.append(copy.deepcopy((params, obfuscation_params)))
        batcher.dump(self.path, override=True)
        for epoch_id in range(params['n_epochs']):
            current_losses = []
            current_regularizers = []
            adversarial_losses = []
            self.ast_encoder.train()
#             self.ast_encoder.eval()
            for code, adversarial_code_list in batcher.train_code(params['train_n_problems'], params['n_adversarial']):
#                 print(adversarial_code_list)
                
#             for iter_id in range(params['n_problems_per_epoch']):
#                 n_users = len(data)
#                 current_user = list(sorted(data.keys()))[np.random.choice(n_users)]
#                 current_data = data[current_user]
#                 cur_len = len(current_data)
#                 code = list(sorted(current_data.keys()))[np.random.choice(cur_len)]
#                 code = batcher.get_train_code()
                cur_loss, cur_reprs, _ = loss_for_problem(self.ast_encoder, code, params['n_obfuscated'], obfuscation_params)
#                 print("OK")
                
                current_losses.append(cur_loss)
                cur_reg = cur_reprs.norm(dim=1).mean()
                current_regularizers.append(cur_reg)
                
                adversarial_loss, _, _ = loss_adversarial(self.ast_encoder, code, adversarial_code_list, obfuscation_params)
                adversarial_losses.append(adversarial_loss)
#                 cur_reprs = cur_reprs.detach().numpy()
#                 sns.heatmap(cosine_similarity(cur_reprs, cur_reprs),  vmin=0.0, vmax=1.0)
#                 plt.show()


            self.optimizer.zero_grad()
            loss = sum(current_losses)/len(current_losses)
            regularizer = torch.abs(1 - sum(current_regularizers)/len(current_regularizers))
            adversarial = sum(adversarial_losses)/len(adversarial_losses)
            real_loss = -loss + params['regularizer_coef'] * regularizer + params['adversarial_coef'] * adversarial
#             real_loss = -loss + params['adversarial_coef'] * adversarial
            real_loss.backward()
            self.optimizer.step()
            self.train_metrics['regularizer'].append(regularizer.detach().item())
            self.train_metrics['loss'].append(loss.detach().item())
            self.train_metrics['adversarial'].append(adversarial.detach().item())
            
            self.dump(os.path.join(self.path, "last_state"), override=True)
            
            self.ast_encoder.eval()
            if epoch_id % params['validate_every'] == 0:
                current_losses = []
                current_regularizers = []
                adversarial_losses = []
#                 for iter_id in range(n_problems_validation):
#                     n_users = len(data)
#                     current_user = list(sorted(data.keys()))[np.random.choice(n_users)]
#                     current_data = data[current_user]
#                     cur_len = len(current_data)
#                     code = list(sorted(current_data.keys()))[np.random.choice(cur_len)]
                for code, adversarial_code_list in batcher.validation_code(params['validate_n_problems'], params['n_adversarial']):
                    cur_loss, cur_reprs, _ = loss_for_problem(self.ast_encoder, code, params['n_obfuscated'], obfuscation_params)
                    current_losses.append(cur_loss)
                    cur_reg = cur_reprs.norm(dim=1).mean()
                    current_regularizers.append(cur_reg)
                    
                    adversarial_loss, _, _ = loss_adversarial(self.ast_encoder, code, adversarial_code_list, obfuscation_params)
                    adversarial_losses.append(adversarial_loss)
            
                loss = sum(current_losses)/len(current_losses)
                regularizer = torch.abs(1 - sum(current_regularizers)/len(current_regularizers))
                adversarial = sum(adversarial_losses)/len(adversarial_losses)
                self.validation_metrics['loss'].append(loss.detach().item())
                self.validation_metrics['iterations'].append(len(self.train_metrics['loss']) - 1)
                self.validation_metrics['regularizer'].append(regularizer.detach().item())
                self.validation_metrics['adversarial'].append(adversarial.detach().item())
                
                self.is_best_state()

            self.plot_all()

def get_n_obfuscated_representations_mixed(ast_encoder, code, code_list, obfuscation_params):
    data_points = []
    obfuscated_list = []
#     print(code)
    for other_code in code_list:
#             print("=================")
#             print(other_code)
            obfuscated = obfuscation.obfuscate_mixed(code, other_code, obfuscation_params)
            obfuscated_list.append(obfuscated)
            obfuscated_repr = ast_encoder(ast.parse(obfuscated)).view(1, -1)
            data_points.append(obfuscated_repr)
    return data_points, obfuscated_list
            
            
def loss_for_problem_mixed(ast_encoder, code, code_list, obfuscation_params):

    obfuscated_repr, obfuscated = get_n_obfuscated_representations_mixed(ast_encoder, code, code_list, obfuscation_params)
    obfuscated_repr = torch.cat(obfuscated_repr, dim=0)
    
    original_repr = ast_encoder(ast.parse(code)).view(1, -1)
    
    original_repr = original_repr.repeat(len(code_list), 1)
    
#     print(cosine_similarity(original_repr.detach(), obfuscated_repr.detach()))
#     print(torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr))
    loss = torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr).mean()
    
    
    return loss, obfuscated_repr, obfuscated

def loss_adversarial_mixed(ast_encoder, code, code_list, obfuscation_params):
    
    
    
    all_repr = []
    all_obfuscated_code = []
    for adversarial_code in code_list:
        obfuscated_repr, obfuscated = get_n_obfuscated_representations_mixed(ast_encoder, adversarial_code, [code], obfuscation_params)

        all_repr += obfuscated_repr
        all_obfuscated_code += obfuscated
    
    
    
    original_repr = ast_encoder(ast.parse(code)).view(1, -1)
    
    if not code_list:
        return 0.0, torch.zeros_like(original_repr), []
    
    original_repr = original_repr.repeat(len(all_repr), 1)

    
                                             
    obfuscated_repr = torch.cat(all_repr, dim=0)
    
    loss = torch.nn.functional.cosine_similarity(original_repr, obfuscated_repr).mean()
    
    
    return loss, obfuscated_repr, obfuscated          
        

            
class MixingTrainer(Trainer):
        
        
    def train(self, batcher, params, obfuscation_params):
        self.all_params.append(copy.deepcopy((params, obfuscation_params)))
        batcher.dump(self.path, override=True)
        for epoch_id in range(params['n_epochs']):
            current_losses = []
            current_regularizers = []
            adversarial_losses = []
            self.ast_encoder.train()
            for code, adversarial_code_list in batcher.train_code(params['train_n_problems'], params['n_adversarial']):
                cur_loss, cur_reprs, _ = loss_for_problem_mixed(self.ast_encoder, code, adversarial_code_list, obfuscation_params)
                
                current_losses.append(cur_loss)
                cur_reg = cur_reprs.norm(dim=1).mean()
                current_regularizers.append(cur_reg)
                
                adversarial_loss, _, _ = loss_adversarial_mixed(self.ast_encoder, code, adversarial_code_list, obfuscation_params)
                adversarial_losses.append(adversarial_loss)
#                 cur_reprs = cur_reprs.detach().numpy()
#                 sns.heatmap(cosine_similarity(cur_reprs, cur_reprs),  vmin=0.0, vmax=1.0)
#                 plt.show()


            self.optimizer.zero_grad()
            loss = sum(current_losses)/len(current_losses)
            regularizer = torch.abs(1 - sum(current_regularizers)/len(current_regularizers))
            adversarial = sum(adversarial_losses)/len(adversarial_losses)
            real_loss = -loss + params['regularizer_coef'] * regularizer + params['adversarial_coef'] * adversarial
#             real_loss = -loss + params['adversarial_coef'] * adversarial
            real_loss.backward()
            self.optimizer.step()
            self.train_metrics['regularizer'].append(regularizer.detach().item())
            self.train_metrics['loss'].append(loss.detach().item())
            self.train_metrics['adversarial'].append(adversarial.detach().item())
            
            self.dump(os.path.join(self.path, "last_state"), override=True)
            
            self.ast_encoder.eval()
            if epoch_id % params['validate_every'] == 0:
                current_losses = []
                current_regularizers = []
                adversarial_losses = []
#                 for iter_id in range(n_problems_validation):
#                     n_users = len(data)
#                     current_user = list(sorted(data.keys()))[np.random.choice(n_users)]
#                     current_data = data[current_user]
#                     cur_len = len(current_data)
#                     code = list(sorted(current_data.keys()))[np.random.choice(cur_len)]
                for code, adversarial_code_list in batcher.validation_code(params['validate_n_problems'], params['n_adversarial']):
                    cur_loss, cur_reprs, _ = loss_for_problem_mixed(self.ast_encoder, code, adversarial_code_list, obfuscation_params)
                    current_losses.append(cur_loss)
                    cur_reg = cur_reprs.norm(dim=1).mean()
                    current_regularizers.append(cur_reg)
                    
                    adversarial_loss, _, _ = loss_adversarial_mixed(self.ast_encoder, code, adversarial_code_list, obfuscation_params)
                    adversarial_losses.append(adversarial_loss)
            
                loss = sum(current_losses)/len(current_losses)
                regularizer = torch.abs(1 - sum(current_regularizers)/len(current_regularizers))
                adversarial = sum(adversarial_losses)/len(adversarial_losses)
                self.validation_metrics['loss'].append(loss.detach().item())
                self.validation_metrics['iterations'].append(len(self.train_metrics['loss']) - 1)
                self.validation_metrics['regularizer'].append(regularizer.detach().item())
                self.validation_metrics['adversarial'].append(adversarial.detach().item())
                
                self.is_best_state()

            self.plot_all()