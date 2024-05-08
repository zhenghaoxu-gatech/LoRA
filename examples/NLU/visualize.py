import os
import json
import matplotlib.pyplot as plt

metric = {
    'mnli': 'eval_accuracy',
    'cola': 'eval_matthews_correlation',
}

def plot_loss(NLU_path, dataset, batch_size, seed=0):
    # Construct path to trainer_state.json
    for init in [None, 'col', 'svd']:
        if init is None: 
            dataset_folder = f'{dataset}_{batch_size}'
        else:
            dataset_folder = f'{dataset}_{init}_{batch_size}'
        if seed > 0:
            dataset_folder = f'{dataset_folder}_{seed}'
        dataset_path = os.path.join(NLU_path, dataset_folder)
        trainer_state_path = os.path.join(dataset_path, 'model', 'trainer_state.json')
        
        # Load trainer state JSON
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # Extract evaluation accuracy from log_history
        log_history = trainer_state['log_history']
        epochs = [entry['epoch'] for entry in log_history if 'loss' in entry]
        evaluation_accuracy = [entry['loss'] for entry in log_history if 'loss' in entry]
        
        # Plot evaluation accuracy over time
        init_name = init if init is not None else 'default'
        plt.plot(epochs[:100], evaluation_accuracy[:100], label=f'{init_name}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'loss over epochs')
    plt.legend()
    plt.show()
    
def plot_evaluation_accuracy(NLU_path, dataset, batch_size, seed=0):
    # Construct path to trainer_state.json
    for init in [None, 'col', 'svd']:
        if init is None: 
            dataset_folder = f'{dataset}_{batch_size}'
        else:
            dataset_folder = f'{dataset}_{init}_{batch_size}'
        if seed > 0:
            dataset_folder = f'{dataset_folder}_{seed}'
        dataset_path = os.path.join(NLU_path, dataset_folder)
        trainer_state_path = os.path.join(dataset_path, 'model', 'trainer_state.json')
        
        # Load trainer state JSON
        with open(trainer_state_path, 'r') as f:
            trainer_state = json.load(f)
        
        # Extract evaluation accuracy from log_history
        log_history = trainer_state['log_history']
        epochs = [entry['epoch'] for entry in log_history if metric[dataset] in entry]
        evaluation_accuracy = [entry[metric[dataset]] for entry in log_history if metric[dataset] in entry]
        
        # Plot evaluation accuracy over time
        init_name = init if init is not None else 'default'
        plt.plot(epochs, evaluation_accuracy, label=f'{init_name}')
    plt.xlabel('epoch')
    plt.ylabel(metric[dataset])
    plt.title(f'{metric[dataset]} over epochs')
    plt.legend()
    plt.show()

# Example usage
base_path = os.getcwd()
NLU_path = os.path.join(base_path, 'examples', 'NLU')
# print(base_path)
dataset = 'mnli'
batch_size = 128

plot_loss(NLU_path, dataset, batch_size, 1)