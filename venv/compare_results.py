import subprocess, csv, sys, re, matplotlib.pyplot as plt

def run(script):
    print(f" Executing {script} (Benchmarking on RTX 5060)")
    res = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Error in {script}:\n{res.stderr}")
        return None
    return res.stdout

def parse_full_data(out):
    """Extracts every epoch line for Loss, Accuracy, and Timing."""
    # Label: Time | Loss: X | Accuracy: Y | Val_Acc: Z
    pattern = r"(.*?): (\d+\.\d+)s \| Loss: (\d+\.\d+) \| Accuracy: (\d+\.\d+)% \| Val_Acc: (\d+\.\d+)%"
    matches = re.findall(pattern, out)
    
    # Structure: (Label, Time, Loss, Acc, Val_Acc)
    return matches

def plot_by_batch(jax_raw, torch_raw):
    """Generates graphs where each batch size"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    batch_labels = ['64', '256', '1024']
    x_axis = range(1, 6) # Epochs 1-5

    for i in range(3):
        start, end = i * 5, (i + 1) * 5
        
        # Extract Loss and Accuracy for the 5-epoch 
        j_loss = [float(m[2]) for m in jax_raw[start:end]]
        j_acc = [float(m[3]) for m in jax_raw[start:end]]
        t_loss = [float(m[2]) for m in torch_raw[start:end]]
        t_acc = [float(m[3]) for m in torch_raw[start:end]]

        # Plot Loss
        ax1.plot(x_axis, j_loss, marker='o', color=colors[i], label=f'JAX (BS:{batch_labels[i]})')
        ax1.plot(x_axis, t_loss, marker='x', color=colors[i], linestyle='--', label=f'Torch (BS:{batch_labels[i]})')
        
        # Plot Accuracy
        ax2.plot(x_axis, j_acc, marker='o', color=colors[i], label=f'JAX (BS:{batch_labels[i]})')
        ax2.plot(x_axis, t_acc, marker='x', color=colors[i], linestyle='--', label=f'Torch (BS:{batch_labels[i]})')

    ax1.set_title("Training Loss: Batch Comparison", fontsize=14)
    ax1.set_xlabel("Epoch Number", fontsize=12); ax1.set_ylabel("Cross-Entropy Loss", fontsize=12)
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Accuracy: Batch Comparison", fontsize=14)
    ax2.set_xlabel("Epoch Number", fontsize=12); ax2.set_ylabel("Accuracy (%)", fontsize=12)
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_curves.png')

if __name__ == "__main__":
    j_path = "/home/sugan/jax_assignment/venv/assignment_1JAX.py"
    t_path = "/home/sugan/jax_assignment/venv/assignment_1TORCH.py"
    
    j_out, t_out = run(j_path), run(t_path)
    
    if j_out and t_out:
        jax_raw = parse_full_data(j_out)
        torch_raw = parse_full_data(t_out)

        # SAVE CSV
        with open('benchmark_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Batch Size", "Framework", "Epoch Label", "Time (s)", "Loss", "Train Acc (%)", "Val Acc (%)"])
            
            for i, b in enumerate([64, 256, 1024]):
                start, end = i * 5, (i + 1) * 5
                # Write JAX block
                for m in jax_raw[start:end]:
                    writer.writerow([b, "JAX", m[0], m[1], m[2], m[3], m[4]])
                # Write PyTorch block
                for m in torch_raw[start:end]:
                    writer.writerow([b, "PyTorch", m[0], m[1], m[2], m[3], m[4]])
        
        plot_by_batch(jax_raw, torch_raw)
        print("DATA SAVED AS CSV. PNG shows overlaid batch curves.")