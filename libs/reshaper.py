

def load_instruction(file):
    """
        Load instructions from file

        Parameters:
            - file: file to load, written in name[string] -> cpp_code[string] csv format

        Returns:
            - an instruction, i.e. a dictionary in name[string] -> cpp_code[string] format
    """
    try:
        import csv
        dict_from_csv = {}
        with open(file, mode='r') as inp:
            reader = csv.reader(inp)
            dict_from_csv = {rows[0]: rows[1] for rows in reader}
        return dict_from_csv
    except Exception:
        import traceback
        print(traceback.print_exc())


def save_instruction(file, object):
    """
        Save instructions to file

        Parameters:
            - file: file to save to
            - object: instruction to save, a dictionary in name[string] -> cpp_code[string] format
    """
    try:
        import csv
        f = csv.writer(open(file, 'w'))
        for key, val in object.items():
            f.writerow([key, val])
    except Exception:
        import traceback
        print(traceback.print_exc())

def tree_reshaper(tree, instruction, save_path, new_tree_name):
    """
        Reshapes a tree according to an instruction

        Parameters:
            - tree: a ROOT.TTree object
            - instruction: an instruction, a dictionary in name[string] -> cpp_code[string] format, 
                that defines tree reshaping
            - save_path: path to save the reshaped tree to
            - new_tree_name: name of the new tree
    """
    import ROOT
    df = ROOT.RDataFrame(tree)
    for name, cpp_code in instruction.items():
        df = df.Define(name, cpp_code)
    df.Snapshot(new_tree_name, save_path, {*instruction.keys()})
