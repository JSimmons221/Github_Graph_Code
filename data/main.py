import ast
import os
from datetime import datetime

import pandas as pd
import pyarrow as pa
from graph_functions import graph_files
import glob
import numpy as np


# A set of commonly used strings as constants so they can be changed if file structure is changed.
CSV = '.csv'
REPO_CSV = '/GeneralData/__Repositories' + CSV
COMMIT_PATH = '/CommitData/C_'
STRUCT_PATH = '/FileStructureData/F_'
ISSUE_PATH = '/IssueData/I_'
TEST = 'C_Alexander-MacDonald_test-repo'
REPO_DATA = "repo_data" + CSV


# Returns the number of open issues, closed issues, and total issues for a given repo
# path is the path to the issue data csv for the repo
def get_issue_data(path):
    df = pd.read_csv(path)
    col = df['state']
    open_issues = col[df.state == 'open'].count()
    closed_issues = col[df.state == 'closed'].count()
    total_issues = col.count()
    return open_issues, closed_issues, total_issues


# Returns the total number of additions and deletions for a given repo
# path is the path to the commit data csv for the repo
def get_commit_data(path):
    df = pd.read_csv(path)
    total_additions = df['totalAdditions'].sum()
    total_deletions = df['totalDeletions'].sum()
    return total_additions, total_deletions


# Returns the number of files for a given repo
# path is the path to the repository file structure csv for the repo
def get_num_files(path):
    df = pd.read_csv(path)
    return df['owner'].count()


# Creates a CSV with all data needed for the machine learning algorithms using the above functions to parse data
# path is the path to where you put the 1_data folder
# output_path is the directory to put the output file in
def read_repositories(path, output_path):
    # filtered_ParsedRepos is in the drive
    repositories = pd.read_csv(r'./data/filtered_ParsedRepos.csv')
    repositories = repositories[['owner', 'repo', 'cloneURL', 'stars', 'dateCreated', 'datePushed', 'numCommits']]
    repositories['dateCreated'] = pd.to_datetime(repositories['dateCreated'], format='%Y-%m-%dT%H:%M:%SZ').astype(
        'int64').divide(10 ** 9).astype('int64')
    repositories['datePushed'] = pd.to_datetime(repositories['datePushed'], format='%Y-%m-%dT%H:%M:%SZ').astype(
        'int64').divide(10 ** 9).astype('int64')

    open_issue_list = []
    closed_issue_list = []
    total_issue_list = []
    additions_list = []
    deletions_list = []
    file_count_list = []
    include = []

    for index, row in repositories.iterrows():
        file_path = row['owner'] + '_' + row['repo'] + CSV
        print(index)
        print(file_path)


        if os.path.isfile(path + ISSUE_PATH + file_path):
            print('succeeded')
            o_issues, c_issues, t_issues = get_issue_data(path + ISSUE_PATH + file_path)
            open_issue_list.append(o_issues)
            closed_issue_list.append(c_issues)
            total_issue_list.append(t_issues)

            total_additions, total_deletions = get_commit_data(path + COMMIT_PATH + file_path)
            additions_list.append(total_additions)
            deletions_list.append(total_deletions)

            file_count = get_num_files(path + STRUCT_PATH + file_path)
            file_count_list.append(file_count)

            include.append(True)

        else:
            print('failed')
            open_issue_list.append(0)
            closed_issue_list.append(0)
            total_issue_list.append(0)
            additions_list.append(0)
            deletions_list.append(0)
            file_count_list.append(0)
            include.append(False)

        print()

    repositories['openIssues'] = open_issue_list
    repositories['closedIssues'] = closed_issue_list
    repositories['totalIssues'] = total_issue_list
    repositories['totalAdditions'] = additions_list
    repositories['totalDeletions'] = deletions_list
    repositories['fileCount'] = file_count_list
    repositories['include'] = include

    repositories = repositories[repositories.include == True]

    repositories = repositories.drop('include', axis=1)
    repositories.to_csv(output_path + REPO_DATA, index=False)


# Calls functions from graph_functions to add a repos data to the DGL dataset
# path is the path to where you put the 1_data folder
# filename is the author and repo name of the repo
# graph_id is an id for the given graph
# output_path is the directory to put the output file in
def get_graph_data(path, filename, graph_id, output_path):
    # Gets this sizes of each of the files in the repository for node features
    file_structure = pd.read_csv(path + STRUCT_PATH + filename)
    file_sizes = file_structure['fileTuple<fileName.fileSize>'].str.split('\\((.+), (.+)\\)', expand=True, regex=True,n=1)
    file_sizes = file_sizes.drop([0, 3], axis=1)

    graph_files(path + COMMIT_PATH + filename, graph_id, file_sizes, output_path)


# Sets up the files for the DGL dataset and calls get_graph_data for each repo
# path is the path to where you put the 1_data folder
# filename is the location of the file from read_repositories
# output_path is the directory to put the output file in
def get_all_data(path, filename, output_path):
    graph_dataframe = pd.read_csv(filename)
    graph_data = graph_dataframe[['owner', 'repo']].to_numpy()
    files = glob.glob(output_path + "/*.csv")
    for f in files:
        os.remove(f)

    f = open(output_path + '/edges.csv', 'w')
    f.write("graph_id,src_id,dst_id,weight\n")
    f.close()

    f = open(output_path + '/nodes.csv', 'w')
    f.write("graph_id,node_id,feats\n")
    f.close()

    f = open(output_path + '/graphs.csv', 'w')
    f.write('graph_id,feats,target\n')

    graph_id = 0
    for repository in graph_data:
        if graph_id % 50 == 0:
            print(graph_id)
            print(datetime.now().strftime("%H:%M:%S"))

        print(repository)
        owner = repository[0]
        repo = repository[1]
        try:
            get_graph_data(path, owner + '_' + repo + CSV, graph_id, output_path)
            f.write(str(graph_id) + '\n')
            graph_id += 1
        except:
            print("An error happened somewhere, I will fix this later to be more exact about what happened (maybe)")

    f.close()


# i_path = "D:/1_Data"
# o_path = "D:/1_Data/GraphData"
# get_all_data(i_path, r'./data/MI_final_parsed.csv', o_path)
