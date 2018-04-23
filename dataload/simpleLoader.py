
                                                                                                                                                           
import os

#Just read file and spilt by newline character
def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')
