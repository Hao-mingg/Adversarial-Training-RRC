import os
import subprocess
import shutil

def run_rrc_python(task, bert, domain, run_dir, runs, cuda_device=None, dropout=0.0, epsilon=0.0):

    if cuda_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    data_dir = os.path.join("..", task, domain)

    # Remove existing directory contents if it exists
    dir = os.path.join("..", "run", run_dir, domain)
    if os.path.exists(dir):
        shutil.rmtree(dir)

    for run in range(1, runs + 1):
        output_dir = os.path.join("..", "run", run_dir, domain, str(run))

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

        train_log_file = os.path.join(output_dir, "train_log.txt")
        valid_file = os.path.join(output_dir, "valid.json")
        predictions_file = os.path.join(output_dir, "predictions.json")
        model_file = os.path.join(output_dir, "model.pt")

        if not os.path.exists(valid_file):
            python_command = [
                "python", "../src/run_rrc.py",
                "--bert_model", domain+"_"+bert, "--do_train", "--do_valid",
                "--gradient_accumulation_steps", "2",
                "--max_seq_length", "320", "--train_batch_size", "16", "--learning_rate", "3e-5",
                "--num_train_epochs", "4",
                "--output_dir", output_dir, "--data_dir", data_dir, "--seed", str(run),
                "--dropout", str(dropout),
                "--epsilon", str(epsilon),
            ]
            with open(train_log_file, "w") as log_file:
                subprocess.run(python_command, stdout=log_file, stderr=subprocess.STDOUT)

        if not os.path.exists(predictions_file):
            python_command = [
                "python", "../src/run_rrc.py",
                "--bert_model", domain+"_"+bert, "--do_eval", "--max_seq_length", "320",
                "--output_dir", output_dir, "--data_dir", data_dir, "--seed", str(run),
            ]
            with open(os.path.join(output_dir, "test_log.txt"), "w") as log_file:
                subprocess.run(python_command, stdout=log_file, stderr=subprocess.STDOUT)

        if os.path.exists(predictions_file) and os.path.exists(model_file):
            os.remove(model_file)