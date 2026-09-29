import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Rectangle


_color_superficial = '#298C8C' # #94DDDE
_color_deep = '#a00000' # #F7B4A7
_color_both = '#B8B8B8' # #CCCCCC

_color_simple = '#384860'


def plot_graph(G: nx.Graph, path_out: str, fig_title: str, style = 'spring'):
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10,10))

    nodelist = G.nodes()
    edgedict = nx.get_edge_attributes(G, 'weight')
    degreedict = dict(G.degree)

    if (style == 'shell'):
        pos = nx.shell_layout(G)
    elif (style == 'kamada_kawai'):
        pos = nx.kamada_kawai_layout(G)
    elif (style == 'circular'):
        pos = nx.circular_layout(G)
    else:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=10)

    electrode_colors = list(nx.get_node_attributes(G, "electrode").values())
    electrode_colors = [_color_superficial if item == 1 else _color_deep for item in electrode_colors]

    nx.draw_networkx_nodes(G, pos, 
                           nodelist = nodelist,
                           node_size = [v * 150 for v in degreedict.values()],
                           node_color = electrode_colors,
                           alpha = 0.5,
                           ax=ax)
    nx.draw_networkx_edges(G, pos,
                           edgelist = edgedict.keys(),
                           width = list(edgedict.values()),
                           edge_color = 'tab:gray',
                           alpha = 0.4,
                           ax=ax)
    nx.draw_networkx_labels(G, pos,
                            labels = dict(zip(nodelist, nodelist)),
                            font_color = 'black',
                            font_weight = 'bold',
                            ax=ax)
    
    ax.set_facecolor('white')    
    ax.grid(False)
    ax.set_title(fig_title, fontsize=20, weight='bold')

    sns.despine(left=True, bottom=True)

    fig.tight_layout()
    fig.savefig(path_out + "_graph.pdf", dpi='figure', bbox_inches="tight")
    plt.close()


def plot_components(G: nx.Graph, path_out: str, fig_title: str, style = 'spring'):
    count = 1
    for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
        if len(C) > 1:
            plot_graph(C, path_out + "_comp" + str(count), fig_title + " Component " + str(count), style)
        count += 1
        

def plot_degrees(G: nx.Graph, path_out: str, fig_title: str):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    list_degrees = [val for (node, val) in G.degree()]
    list_edge_weights = list(nx.get_edge_attributes(G,'weight').values())
    
    ax[0].set_title("Degree distribution")
    ax[0].hist(list_degrees, alpha=0.8, density=True, label="Degree", color=_color_simple)
    ax[0].set_xlabel("Degree")
    ax[0].set_ylabel("Density")
    
    ax[1].set_title("Edge weight distribution")
    ax[1].hist(list_edge_weights, alpha=0.8, density=True, label="Weights", color=_color_simple)
    ax[1].set_ylabel("Density")
    ax[1].set_xlabel("Edge weight")

    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(path_out + "_graph_degree.pdf", dpi='figure', bbox_inches="tight")
    plt.close()

def plot_neuron_properties(df_neurons: pd.DataFrame, path_out: str, fig_title: str):
    fig, ax = plt.subplots(2, 2, figsize=(10,10))

    palette = {1: _color_superficial, 2: _color_deep}
    
    # ax[0][0].set_title("Firing rate distribution")
    # # ax[0][0].hist((df_neurons[(df_neurons['Electrode']==1)])['Firing Rate'], alpha=0.8, density=True, label="Superficial", color=_color_superficial)
    # # ax[0][0].hist((df_neurons[(df_neurons['Electrode']==2)])['Firing Rate'], alpha=0.8, density=True, label="Deep", color=_color_deep)
    # sns.histplot(df_neurons, x="Firing Rate", hue="Electrode", legend=False, ax=ax[0][0], palette=palette, element="step", log_scale=True)
    # ax[0][0].set_xscale('log')
    # ax[0][0].set_xlabel("Firing rate")
    # ax[0][0].set_ylabel("Density")
    # ax[0][0].legend(title="Location", labels=['Superficial', 'Deep'])
    
    ax[0][1].set_title("Clustering")
    # ax[0][1].hist((df_neurons[(df_neurons['Electrode']==1)])['Clustering'], alpha=0.8, density=True, label="Superficial", color=_color_superficial)
    # ax[0][1].hist((df_neurons[(df_neurons['Electrode']==2)])['Clustering'], alpha=0.8, density=True, label="Deep", color=_color_deep)
    sns.histplot(df_neurons, x="Clustering", hue="Electrode", legend=False, ax=ax[0][1], palette=palette, element="step")
    ax[0][1].set_xlabel("Clustering coefficient")
    ax[0][1].set_ylabel("Density")
    ax[0][1].legend(title="Location", labels=['Superficial', 'Deep'])

    ax[1][0].set_title("Betweenness Centrality")
    # ax[1][0].hist((df_neurons[(df_neurons['Electrode']==1)])['Betweenness Centrality'], alpha=0.8, density=True, label="Superficial", color=_color_superficial)
    # ax[1][0].hist((df_neurons[(df_neurons['Electrode']==2)])['Betweenness Centrality'], alpha=0.8, density=True, label="Deep", color=_color_deep)
    sns.histplot(df_neurons, x="Betweenness Centrality", hue="Electrode", legend=False, ax=ax[1][0], palette=palette, element="step")
    ax[1][0].set_xlabel("Betweenness Centrality")
    ax[1][0].set_ylabel("Density")
    ax[1][0].legend(title="Location", labels=['Superficial', 'Deep'])
    
    ax[1][1].set_title("Norm. Hub Value")
    # ax[1][1].hist((df_neurons[(df_neurons['Electrode']==1)])["Norm. Hub Value"], alpha=0.8, density=True, label="Superficial", color=_color_superficial)
    # ax[1][1].hist((df_neurons[(df_neurons['Electrode']==2)])["Norm. Hub Value"], alpha=0.8, density=True, label="Deep", color=_color_deep)
    sns.histplot(df_neurons, x="Norm. Hub Value", hue="Electrode", legend=False, ax=ax[1][1], palette=palette, element="step")
    ax[1][1].set_xlabel("Norm. Hub Value")
    ax[1][1].set_ylabel("Density")
    ax[1][1].legend(title="Location", labels=['Superficial', 'Deep'])

    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(path_out + "_df_neurons.pdf", dpi='figure', bbox_inches="tight")
    plt.close()

def plot_edge_properties(df_edges: pd.DataFrame, path_out: str, fig_title: str):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    palette = {'Superficial': _color_superficial, 'Deep': _color_deep, 'Crosslayer': _color_both}
    
    ax[0].set_title("Edge weight distribution")
    # ax[0].hist(df_edges['Weight'], alpha=0.8, density=True, label="Weights", color=_color_simple)
    sns.histplot(df_edges, x="Weight", color=_color_simple, ax=ax[0])
    ax[0].set_ylabel("Density")
    ax[0].set_xlabel("Edge weight")
    
    ax[1].set_title("Edge weight distribution by location")
    sns.histplot(df_edges, x="Weight", hue="Location", palette=palette, element="step", ax=ax[1])
    # ax[1].hist((df_edges[(df_edges['Location']=="Superficial")])['Weight'], alpha=0.8, density=True, label="Superficial", color=_color_superficial)
    # ax[1].hist((df_edges[(df_edges['Location']=="Deep")])['Weight'], alpha=0.8, density=True, label="Deep", color=_color_deep)
    # ax[1].hist((df_edges[(df_edges['Location']=="Both")])['Weight'], alpha=0.8, density=True, label="Both", color=_color_both)
    ax[1].set_ylabel("Density")
    ax[1].set_xlabel("Edge weight")
    # ax[1].legend()

    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(path_out + "_df_edges.pdf", dpi='figure', bbox_inches="tight")
    plt.close()

# def plot_reward_properties(df_reward: pd.DataFrame, path_out: str, fig_title: str):
#     fig, ax = plt.subplots(1, 2, figsize=(10,5))

#     avg_count = { 'pos': 0, 'neg': 0 }
#     custom_palette = {}
#     for idx in set(df_reward["Profile ID"]):
#         avg = (np.average(df_reward[df_reward["Profile ID"] == idx]["Time Distance (ms)"]))
#         if avg < 0:
#             custom_palette[idx] = 'tab:red'
#             avg_count['neg'] += 1
#         else:
#             custom_palette[idx] = 'tab:cyan'
#             avg_count['pos'] += 1

#     sns.pointplot(df_reward, x="Time Distance (ms)", y="Profile ID", estimator='mean', errorbar='sd') #, linestyle="none", orient='h', ax=ax[0], hue="Profile ID", palette=custom_palette)
#     anchored_text = AnchoredText("Positive: "+str(avg_count['pos'])+"\n"+"Negative: "+str(avg_count['neg']), loc=2)
#     ax[0].add_artist(anchored_text)

#     fig.suptitle(fig_title)
#     fig.tight_layout()
#     fig.savefig(path_out + "_df_reward.pdf", dpi='figure', bbox_inches="tight")
#     plt.close()

def plot_reward_properties(df_reward: pd.DataFrame, path_out: str, fig_title: str):
    fig, ax = plt.subplots(1, 2, figsize=(10,10))

    df_reward_success = df_reward.loc[df_reward['Event'] == 'Success'].copy()
    df_reward_failure = df_reward.loc[df_reward['Event'] == 'Failure'].copy()

    colors = [['#83B7DF','#036D9C'],['#EF8986', '#AF2138']]

    for i, df in enumerate([df_reward_success, df_reward_failure]):
        # Converts 'Profile ID' to string to avoid issues with sns.pointplot
        df['Profile ID'] = df['Profile ID'].astype(str)
        num_profiles = len(set(np.unique(df["Profile ID"])))

        if num_profiles > 1:
            avg_count = { 'pos': 0, 'neg': 0 }
            custom_palette = {}
            for idx in set(df["Profile ID"]):
                avg = (np.average(df[df["Profile ID"] == idx]["Time Distance (ms)"]))
                if avg < 0:
                    custom_palette[idx] = colors[i][0]
                    avg_count['neg'] += 1
                else:
                    custom_palette[idx] = colors[i][1]
                    avg_count['pos'] += 1
            
            ax[i].vlines(0, 0, num_profiles+1, 'black', linewidth=2)
            ax[i].add_patch(Rectangle((-1000, 0), 1000, num_profiles+1, facecolor='gray', alpha=0.1, hatch='//'))
            ax[i].vlines(-1000, 0, num_profiles+1, 'gray', ':', linewidth=2)
            ax[i].set_xlim(-1500, 1500)

            ax[i].set_ylim(0, num_profiles)
            
            mean_values = df.groupby('Profile ID')['Time Distance (ms)'].mean().sort_values()
            ordered_profile_ids = mean_values.index.astype(str) # convert to string if necessary

            sns.pointplot(data=df, x="Time Distance (ms)", y="Profile ID", 
                        estimator=np.mean, errorbar='sd', linestyle="none", orient='h', 
                        hue="Profile ID", palette=custom_palette, 
                        order=ordered_profile_ids, ax=ax[i])

            ax[i].get_legend().set_visible(False)
            ax[i].set_yticklabels([])
            ax[i].set_xticks(np.arange(-1500, 1501, 500))

            anchored_text = AnchoredText("Positive: "+str(avg_count['pos'])+"\n"+"Negative: "+str(avg_count['neg']), loc=1)
            ax[i].add_artist(anchored_text)

            ax[i].set_xlabel("Time from pull movement (ms)")

    ax[0].vlines(200, 0, num_profiles+1, 'gray', '--', linewidth=2)

    ax[0].set_ylabel("Profiles ordered by mean time distance to pull movement (successful trials)")
    ax[1].set_ylabel("Profiles ordered by mean time distance to pull movement (failed trials)")

    sns.despine(left=True, bottom=False)

    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(path_out + "_df_reward.pdf", dpi='figure', bbox_inches="tight")
    plt.close()

def plot_event_alignment(df_reward: pd.DataFrame, path_out: str, fig_title: str, interval: int = 1500):
    df_reward = df_reward.loc[df_reward['Event'] != 'Success_Signal'].copy() # remove success signal events
                              
    fig, ax = plt.subplots(1, 4, figsize=(20,5))

    df_reward_success = df_reward.loc[df_reward['Event'] == 'Success'].copy()
    df_reward_failure = df_reward.loc[df_reward['Event'] == 'Failure'].copy()

    for i in range(3):
        ax[i].set_xlim(-interval, interval)

    ax[2].vlines(0, 0, 1, 'black', linewidth=2)
    ax[2].vlines(-1000, 0, 1, 'gray', '--', linewidth=2)
    ax[2].vlines(200, 0, 1, 'gray', ':', linewidth=2)

    bins, edges = np.histogram(df_reward_success['Time Distance (ms)'], bins=30, density=False, range=(-interval, interval))
    ax[0].bar(edges[:-1], bins, width=np.diff(edges), color='#0D95D0', label="Success")
    max_height_success = max(bins)

    bins, edges = np.histogram(df_reward_failure['Time Distance (ms)'], bins=30, density=False, range=(-interval, interval))
    ax[1].bar(edges[:-1], bins, width=np.diff(edges), color='#E72F52', label="Failure")
    max_height_failure = max(bins)

    max_height = max(max_height_success, max_height_failure)

    for i in range(2):
        ax[i].set_ylim(0, max_height)
        ax[i].vlines(0, 0, max_height, 'black', linewidth=2)
        ax[i].vlines(-1000, 0, max_height, 'gray', '--', linewidth=2)
    
    ax[0].vlines(200, 0, max_height, 'gray', ':', linewidth=2)

    # sns.histplot(data=df_reward_success, x='Time Distance (ms)', ax=ax[0], color='tab:cyan', bins=30, stat='density')
    # sns.histplot(data=df_reward_failure, x='Time Distance (ms)', ax=ax[1], color='tab:red', bins=30, stat='density')
    sns.ecdfplot(data=df_reward, x='Time Distance (ms)', hue='Event', ax=ax[2], palette=['#0D95D0', '#E72F52'])           
    ax[2].hlines(0.5, -interval, interval, 'black', linewidth=1)

    ax[0].set_ylabel("Profile occurrence (successful trials)")
    ax[1].set_ylabel("Profile occurrence (failed trials)")
    ax[0].set_xlabel("Time from pull movement (ms)")
    ax[1].set_xlabel("Time from pull movement (ms)")

    ax[2].set_xlabel("Cumulative distribution")
    ax[2].set_xlabel("Time from pull movement (ms)") 

    sns.histplot(df_reward, x='Event Time (ms)', hue='Event', ax=ax[3], palette=['tab:cyan', 'tab:red'], bins=30, element="poly")

    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(path_out + "_event_alignment.pdf", dpi='figure', bbox_inches="tight")
    plt.close()

def plot_lever_dynamics(df_reward: pd.DataFrame, lever_data: np.ndarray, path_out: str, fig_title: str, interval: int = 1500):
    fig, ax = plt.subplots(1, 2, figsize=(10,5))

    df_reward_success = df_reward.loc[df_reward['Event'] == 'Success'].copy()
    reward_times = np.unique(df_reward_success['Event Time (ms)'])

    df_reward_failure = df_reward.loc[df_reward['Event'] == 'Failure'].copy()
    failure_times = np.unique(df_reward_failure['Event Time (ms)'])

    for idx, event_times in enumerate([reward_times, failure_times]):
        ax[idx].vlines(0, 0, 1, 'black', linewidth=2)
        ax[idx].vlines(-1000, 0, 1, 'gray', '--', linewidth=2)
        ax[idx].set_xlim(-interval, interval)        
        ax[idx].set_ylim(0, 1)

        colors = ['#0D95D0', '#E72F52']
        for event in event_times:    
            local_time = lever_data[0][(lever_data[0] > event - interval) & (lever_data[0] < event + interval)] - event
            local_value = lever_data[1][(lever_data[0] > event - interval) & (lever_data[0] < event + interval)]
            ax[idx].plot(local_time, local_value, colors[idx], alpha=0.1)

    ax[0].vlines(200, 0, 1, 'gray', ':', linewidth=2)
    ax[0].set_ylabel("Normalized lever position (successful trials)")
    ax[1].set_ylabel("Normalized lever position (failed trials)")
    ax[0].set_xlabel("Time from pull movement (ms)")
    ax[1].set_xlabel("Time from pull movement (ms)")            

    fig.suptitle(fig_title)
    fig.tight_layout()
    fig.savefig(path_out + "_lever.pdf", dpi='figure', bbox_inches="tight")
    plt.close()