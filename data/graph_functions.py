import ast
import json

import pandas as pd


# Determines the nodes of the graphs and creates lists of which nodes appear in which commits for graphing purposes
# fname is the path to the commit_data csv for the repo
# graph_id is the id for the graph in the DGL dataset
# file_sizes is a data frame with the files their file sizes for a repo
# files is an empty dictionary to store the files and their data in
def get_repo_graph(fname, graph_id, file_sizes, files):
    df = pd.read_csv(fname)
    # Old column title was unwieldy
    df = df.rename(columns={'fileTuples<fileName. status. additions. deletions. changes. raw_url. contents_url>': 'files'})
    df = df.fillna(value='N/A')
    df["files"] = df["files"].apply(ast.literal_eval)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H:%M:%SZ').astype(
        'int64').divide(10 ** 9).astype('int64')

    commits_w_info = df[['date', 'files']].to_numpy()
    commits = []
    node_id = 0

    # Opens the langs json which is used to determine what file type a file is
    with open("langs.json", 'r') as start:
        hard_coded = json.load(start)
        start.close()

    # Loop through each of the sets of commit info, i is used to limit the number of commits to 100
    i = 0
    for commit_info in commits_w_info:
        # Retrieve the commit date and file list
        date = commit_info[0]
        commit_files = commit_info[1]

        # If a commit only has one file in it, the commit would create no edges and thus can be ignored
        if len(commit_files) > 1:
            i += 1
            file_paths = []
            for file in commit_files:
                # Check to see if the file has been found on previous iterations, if it hasn't, add it and it's info, otherwise continue
                file_path = "'main/" + file[0] + "'"
                if file_path not in files:
                    intermediate = file_sizes[file_sizes[1] == file_path]
                    file_size = 0
                    if intermediate.shape[0] == 1:
                        file_size = int(file_sizes[file_sizes[1] == file_path].iloc[0][2])

                    extension = file_path[file_path.rfind("."):-1]
                    file_type = get_file_lang(extension, hard_coded)
                    features = [file_size, date] + file_type
                    file_data = [graph_id, str(node_id), features]
                    files.update({file_path: file_data})
                    node_id += 1

                # Add the file to the list of files for this commit
                file_paths.append(file_path)

            commits.append(file_paths)
        if i == 100:
            break

    return commits, node_id


# Creates the data for a repository to put in the DGL dataset
# fname is the path to the commit_data csv for the repo
# graph_id is the id for the graph in the DGL dataset
# file_sizes is a data frame with the files their file sizes for a repo
# output_path is where the DGL dataset is being stored
def graph_files(fname, graph_id, file_sizes, output_path):
    files = {}
    commit_files, node_id = get_repo_graph(fname, graph_id, file_sizes, files)

    g = {}
    for commit in commit_files:
        nodes = []
        # Get the node ids for this commit
        for file in commit:
            nodes.append(files.get(file)[1])

        # Loop through every pair of files in the commit and add/update the edge between the two files.
        for i in range(0, len(nodes) - 1):
            for j in range(i + 1, len(nodes)):
                f_i = nodes[i]
                f_j = nodes[j]
                edge = f_i + "," + f_j
                if edge in g:
                    edge_data = g.get(edge)
                    edge_data[3] = edge_data[3] + 1
                    g.update({edge: edge_data})

                else:
                    edge_data = [graph_id, f_i, f_j, 1]
                    g.update({edge: edge_data})

    nodes = pd.DataFrame.from_dict(files, orient='index')
    nodes.to_csv(output_path + '/nodes.csv', header=False, mode='a', index=False)

    edges = pd.DataFrame.from_dict(g, orient='index')
    edges.to_csv(output_path + '/edges.csv', header=False, mode='a', index=False)


# Returns the encoding of a language type based on an extension
# extension is a file extension
# hard_coded is a loaded .json file
def get_file_lang(extension, hard_coded):
    if extension in hard_coded:
        return hard_coded[extension]
    else:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

