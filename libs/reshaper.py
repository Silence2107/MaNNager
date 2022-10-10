

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


def dataframe_reshaper(tree, instruction, df_range=None, intermediate_tree_save_path=None, vectorization=True):
    """
        Reshapes a tree according to an instruction, resulting in pd.DataFrame

        Parameters:
            - tree: a ROOT.TTree object
            - instruction: an instruction, a dictionary in name[string] -> cpp_code[string] format, 
                that defines tree reshaping
            - save_path: path to save the resulting dataframe to
            - df_range: an iterable of form (begin, end, stride=1). If applied, reshaping is only performed on specified range
            - intermediate_tree_save_path: reshaped tree save path. If not specified, intermediate tree is discarded
            - vectorization: if True, performs checks whether there are columns with vector types, effectively replacing them with numpy.arrays.
                Disabling this feature is safe if you do not have vectorized data, otherwise behaviour is not well defined.

        Returns:
            - a pd.DataFrame, filled in accordance to instruction
    """
    import ROOT
    import pandas as pd
    rdf = ROOT.RDataFrame(tree)
    for name, cpp_code in instruction.items():
        rdf = rdf.Define(name, cpp_code)

    if not df_range == None:
        rdf = rdf.Range(*df_range)
    if not intermediate_tree_save_path == None:
        rdf.Snapshot('reshaped_tree', intermediate_tree_save_path,
                     {*instruction.keys()})

    df = rdf.AsNumpy(columns=[*instruction.keys()])

    if vectorization:
        import numpy as np
        for column in df.keys():
            # check if column is vector-like
            if len(np.array([df[column][0]]).shape) > 1:
                # convert to numpy array
                for index in range(len(df[column])):
                    df[column][index] = np.array([df[column][index]])
    
    df = pd.DataFrame(df)

    return df
