import os
import zipfile
import sys

FILES_COMMON = ["main.py", "test_pred.csv", "hw3_report.pdf", "preprocess.py", "model.py", "train_model.py","gru.py","rnn.py","LSTM_CNN.py"]
TOML = "pyproject.toml"
REQS = "requirements.txt"

def main():
    # Determine dependency file
    if os.path.isfile(TOML):
        dep_file = TOML
    elif os.path.isfile(REQS):
        dep_file = REQS
    else:
        print("Error: No dependencies file found. Provide pyproject.toml or requirements.txt.")
        sys.exit(1)

    # Collect all files to add
    files_to_zip = FILES_COMMON + [dep_file]

    # Create zip
    zip_name = "hw3_submission.zip"
    with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files_to_zip:
            if os.path.isfile(f):
                z.write(f)
                print(f"Added: {f}")
            else:
                print(f"Warning: {f} not found!")

    print(f"\nCreated {zip_name} successfully.")

if __name__ == "__main__":
    main()
