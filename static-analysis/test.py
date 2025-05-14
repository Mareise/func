# processor.py
import sys
import os


if __name__ == "__main__":
    print("This is the static analysis script.")
    # print the current working directory
    print("Current working directory:", os.getcwd())
    # print every file in the current directory
    print("Files in the current directory:")
    for file in os.listdir(os.getcwd()):
        print(file)

