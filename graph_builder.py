#%% Imports 
# General
from collections import Counter
from itertools import combinations, product, permutations
from difflib import SequenceMatcher
from typing import Union
import re, os, json, math

# Data science
import pandas as pd
from pandas import DataFrame
import numpy as np
import networkx as nx
from scipy import stats

# Internal imports
from internals import importData, LOG

#%% Constants


#%% Classes & Functions


class GraphBuilder:
    """
    Takes GPT-3 output triples and builds graph GPT-3 output 
    
    Takes structured statements and generates a NetworkX graph depending 
    on the method 
    """
    def __init__(self):
        self.graph = nx.Graph()
        self.node_counters: dict[str, Counter[str]] = dict()
        self.edge_counters: dict[tuple[str, str], Counter[tuple[str, str]]] = dict()
        self.df_root_name: str = "" # Populated by the root name of the last df that was used to populate counters
        self.graph_root_name: str = "" # Derived from df_root_name, includes information about thresholding


    def popCountersMulti(self, df_path, 
                         col = "Processed_ents",
                         col_sub = "", 
                         subset: Union[str, int] = "",
                         col_sampsize = "",
                         intra_type = False,
                         ):
        """
        For DFs containing multiple entity types (i.e., distinguishes between node types for node and edge enties)
        Populates the graph's counters using df and abbreviation container originally passed 
        into the class 
        DOES NOT RESET THE COUNTERS
        intra_type - whether edges should be created between entities of the same type 
        col_sub and subset arguments used to get subset of df (e.g., certain topics)
        col_sampsize for sample size
        """
        self.df_root_name = os.path.splitext(df_path)[0] # Store root name
        df = importData(df_path, screen_text=[col])
        if col_sub and subset != "": # Need to specify '!= ""' to allow passing of 0 as a subset
            df = df[df[col_sub] == subset]
        proto_stmt: dict[str, list[str]] = json.loads(df[col].iloc[0])[0] # Get prototypical statement (obtains first statment from first row of col of interest)
        ent_types = [key for key in proto_stmt]
        edge_types = list(permutations(proto_stmt, 2))
        LOG.info("Populating counters from DF...") # Below code runs quite fast so we don't need to track progress
        for index, row in df.iterrows():
            sample_size = row[col_sampsize] if col_sampsize else 1
            list_statements: list[dict[str, list[str]]] = json.loads(row[col])
            article_nodes: dict[str, set[str]] = {t: set() for t in ent_types} # Initialize node container
            article_edges: dict[tuple[str, str], set[tuple[str, str]]] = {(t[0], t[1]): set() for t in edge_types} # Initialize edge container for all types
            if intra_type:
                article_edges.update({(et, et): set() for et in ent_types}) # Add container for edges between the same ent types
            
            for statement in list_statements:
                for ent_type in statement: # Parsing for each type of entity type
                    ents = statement[ent_type]
                    article_nodes[ent_type].update(ents) # Add ent to corresponding ent tracker set
                for ent_type1, ent_type2 in combinations(statement, 2):
                    # Can add if statement to screen out relationship between certain types of nodes 
                    for ent1, ent2 in product(statement[ent_type1], statement[ent_type2]):
                        article_edges[(ent_type1, ent_type2)].add((ent1, ent2))
                        article_edges[(ent_type2, ent_type1)].add((ent2, ent1)) # Add reverse relationship 
                if intra_type:
                    for ent_type in statement:
                        ents = statement[ent_type]
                        for ent1, ent2 in combinations(ents, 2):
                            article_edges[(ent_type, ent_type)].add((ent1, ent2))
                            article_edges[(ent_type, ent_type)].add((ent2, ent1)) # Add reverse relationship 
                            
            for ent_type in article_nodes:
                if ent_type not in self.node_counters: # Instantiate node counter if not already instantiated
                    self.node_counters[ent_type] = Counter()
                ent_type_counter = self.node_counters[ent_type] # Retrieve ent type's respective counter
                for ent in article_nodes[ent_type]:
                    ent_type_counter[ent] += sample_size # Scale by sample size
            for edge_type in article_edges:
                if edge_type not in self.edge_counters: # Instantiate edge type counter if it doesn't exist
                    self.edge_counters[edge_type] = Counter()
                edge_type_counter = self.edge_counters[edge_type] # Retrieve edge type's respective counter
                for edge in article_edges[edge_type]:
                    edge_type_counter[edge] += sample_size

    def printCounters(self):
        LOG.info("Entities ==============================================")
        for ent_type in self.node_counters:
            LOG.info(F"------------ Entity type {ent_type} ------------")
            list_items = list(self.node_counters[ent_type].items())
            list_items.sort(key=lambda x: x[1], reverse=True)
            LOG.info(list_items)
            # Print nodes sorted by count
        LOG.info("Edges ==============================================")
        for edge_type in self.edge_counters:
            LOG.info(F"------------ Edge type {edge_type} ------------")
            list_items = list(self.edge_counters[edge_type].items())
            list_items.sort(key=lambda x: x[1], reverse=True)
            LOG.info(list_items)
            # Print edges sorted by count

    
    def buildGraph(self, thresh = 1, multidi = True):
        """
        Builds Networkx graph with the populated counters and a threshold for node count
        ---
        thresh: lower threshold for number of counts needed for each node (exclusive)
        """
        # Reminder that nx nodes can have abitrary attributes that don't contribute to rendering, need to manually adjust visual parameters with drawing methods
        # nx.Graph is just a way to store data, data can be stored in node attributes         
        if multidi:
            self.graph = nx.MultiDiGraph() # Instantiate multidigraph
            # Allows parallel and directed relationships to be rendered
        else:
            self.graph = nx.Graph() # Instantiate regular graph
        LOG.info("Building graph from counters...")
        for ent_type in self.node_counters:
            node_counter = self.node_counters[ent_type]
            
            # Can add specific data into into node depending on node type
            if ent_type == "factor":
                pass
            elif ent_type == "outcome":
                pass
            else: 
                pass 
                
            for ent in node_counter:
                count = node_counter[ent]
                if count > thresh:
                    self.graph.add_node(ent, size=count, ent_type=ent_type)
                pass
        
        for edge_type in self.edge_counters:
            edge_counter = self.edge_counters[edge_type]
            node_s_counter = self.node_counters[edge_type[0]] # Retrieve node counter for source
            node_t_counter = self.node_counters[edge_type[1]] # Retrieve node counter for target
            
            if edge_type == ("factor", "outcome"): # Demo for different styling
                pass
            elif edge_type == ("outcome", "factor"):
                pass
            elif "factor" in edge_type and "outcome" in edge_type: # Can also find bi-directional relationships
                pass
            
            for edge in edge_counter:
                count = edge_counter[edge]
                node_s = edge[0]
                node_t = edge[1]
                if (node_s_counter[node_s] > thresh and node_t_counter[node_t] > thresh): # Don't need to check in both counters like before since node type is already specified by edge_type
                    node_s_mentions = node_s_counter[node_s]
                    probability = count/node_s_mentions # Probability of this connection is number of articles supporting this connection divided by total number of articles mentioning source
                    self.graph.add_edge(node_s, node_t, width=count, prob=probability) # "width" attribute affects pyvis rendering, pyvis doesn't support edge opacity
        self.graph_root_name = self.df_root_name + f"_t{thresh}" # Add threshold information 

    def buildGraphLogistic(self, df_path, cols_logistical, col_sub = None, 
                           subset = None, multidi: bool = True):
        """
        Takes df with rows representing observations and columns representing factors with 
        each cell having 0 or 1 indicator presence of factor
        ---
        thresh: lower threshold for number of counts needed for each node (exclusive)
        """
        # Reminder that nx nodes can have abitrary attributes that don't contribute to rendering, need to manually adjust visual parameters with drawing methods
        # nx.Graph is just a way to store data, data can be stored in node attributes         
        if multidi:
            self.graph = nx.MultiDiGraph() # Instantiate multidigraph
            # Allows parallel and directed relationships to be rendered
        else:
            self.graph = nx.Graph() # Instantiate regular graph

        self.df_root_name = os.path.splitext(df_path)[0] # Store root name
        self.graph_root_name = self.df_root_name + f"_logi"
        df = importData(df_path)
        
        if col_sub and subset: # If both arguments are given for subset, then filter by subset
            self.graph_root_name = self.df_root_name + f"_logi_{col_sub}_{subset}"
            df = df[df[col_sub] == subset]

        # Calculations
        
        total_n = len(df)

        nodes = {column: df[column].sum() for column in cols_logistical if df[column].sum() > 0} # Dict comprehension to get sums of each column
        for node in nodes:
            count = nodes[node]
            self.graph.add_node(node, size = count, ent_type = "column")
        
        
        edges = Counter()
        for index, row in df.iterrows(): # Count edges 
            nodes_row = [] # Store list of nodes in this row # Shouldn't have any duplicates
            for column in cols_logistical:
                if row[column] == 1:
                    nodes_row.append(column)
            for node_start, node_end in combinations(nodes_row, 2):
                edges[(node_start, node_end)] += 1 # Bidirectional connection
                edges[(node_end, node_start)] += 1
            
        for node_s, node_t in edges:
            count = edges[(node_s, node_t)]
            
            node_s_mentions = nodes[node_s]
            node_t_mentions = nodes[node_t]
            max_n_mentions = max(node_s_mentions, node_t_mentions)
            probability = count/node_s_mentions # Probability of this connection is number of articles supporting this connection divided by total number of articles mentioning source
            # if node_s_mentions >= 0.1 * total_n and node_t_mentions >= 0.1 * total_n: # Only append edge if counts are greater than 10% of total (to reduce noise)
            
            
            
            if 1: # Using relative risk instead            
                rr = (count * total_n) / (nodes[node_s] * nodes[node_t]) # https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000353
                p_corr = ((count * total_n) - (nodes[node_s] * nodes[node_t])) / (nodes[node_s] * nodes[node_t] * (total_n - nodes[node_s]) * (total_n - nodes[node_t])) ** 0.5
                t_stat = (p_corr * (total_n - 2) ** 0.5) / (1 - p_corr ** 2) ** 0.5 # n=max(Pi,Pj) << N, which represents the most stringent way in which t can be calculated given our data, as using n=N will produce a larger number of significant links
                t_stat = (p_corr * (max_n_mentions - 2) ** 0.5) / (1 - p_corr ** 2) ** 0.5 # n=max(Pi,Pj) << N, which represents the most stringent way in which t can be calculated given our data, as using n=N will produce a larger number of significant links
                
                sd_rr = (1 / count) + (1 / (nodes[node_s] * nodes[node_t])) - (1 / total_n) - (1 / total_n ** 2) # Real formula for RR distribution
                t_stat = (1 - abs(rr))/sd_rr
                
                p_val = stats.t.sf(np.abs(t_stat), max_n_mentions - 1)
                
                LOG.info(F"{node_s} | {node_t} | RR: {rr} ({rr * math.exp(-1.96 * sd_rr)} - {rr * math.exp(1.96 * sd_rr)}) | Pcorr: {p_corr} | Pval: {p_val}")
                
                if rr > 1.5 and p_val < 0.05: # Filter for only positive associations and significance by t_stat
                    self.graph.add_edge(node_s, node_t, width=count, prob=rr, prob2=rr**2)
            else: # Otherwise use raw association 
                self.graph.add_edge(node_s, node_t, width=count, prob=probability, prob2=probability**2)
                
        return None

    def exportGraph(self, path: Union[str, bytes, os.PathLike] = ""):
        """
        Exports currently stored graph to an XML file with the specified path 
        """
        if path:
            file_name = path
        else:
            file_name = self.graph_root_name + ".xml"
        # Export graph , https://networkx.org/documentation/stable/reference/readwrite/generated/networkx.readwrite.graphml.write_graphml.html#networkx.readwrite.graphml.write_graphml
        # Graphml documentation: https://networkx.org/documentation/stable/reference/readwrite/graphml.html
        nx.write_graphml(self.graph, file_name)
        LOG.info(F"Exported graph to {file_name}")
        
    def resetCounters(self):
        self.node_counters: dict[str, Counter[str]] = dict()
        self.edge_counters: dict[tuple[str, str], Counter[tuple[str, str]]] = dict()

    def resetGraph(self):
        self.graph = nx.Graph()

if __name__ == "__main__": # For testing purposes
    a = GraphBuilder()
    a.buildGraphLogistic("data/tbi_elix_btlabels.xlsx")
    a.exportGraph("data/comorb_probscaled.xml")
    
    # b = GraphBuilder()
    # topic = 0
    # thresh = 1
    # b.popCountersMulti(f"data/gpt3_output_gpt3F_entsF_topics.xlsx",
    #                    col_sub="Topic", subset=topic)
    # b.buildGraph(thresh=thresh, multidi=True)
    # b.exportGraph(f"data/gpt3_output_gpt3F_entsF_topics{topic}_t{thresh}.xml")

