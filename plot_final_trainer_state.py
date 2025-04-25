import os
import json
import matplotlib.pyplot as plt

def plot_training_curves(json_filename='trainer_state.json'):
    """
    Plots training loss, eval loss, and eval accuracy from a HuggingFace trainer_state.json file
    located in the current directory.
    
    Args:
        json_filename (str): Name of the JSON file. Default is 'trainer_state.json'.
    """

    # Path to JSON assuming it lives in the current working directory
    cwd = os.getcwd()
    json_path = os.path.join(cwd, json_filename)

    # Step 1: Load your JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Step 2: Extract the log history
    logs = data['log_history']

    # Step 3: Separate out the metrics
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    eval_accuracies = []

    for entry in logs:
        if 'loss' in entry:  # training step
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry:  # eval step
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
            eval_accuracies.append(entry['eval_accuracy'])

    # Step 4: Plot
    plt.figure(figsize=(14, 6))

    # Plot Training and Eval Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_losses, label='Training Loss')
    plt.plot(eval_steps, eval_losses, label='Eval Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Eval Loss')
    plt.legend()
    plt.grid(True)

    # Plot Eval Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(eval_steps, eval_accuracies, label='Eval Accuracy', color='green')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Eval Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Step 5: Save the plot in the same directory as the JSON
    plot_filename = "training_plot.png"
    plot_path = os.path.join(cwd, plot_filename)

    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_training_curves()
