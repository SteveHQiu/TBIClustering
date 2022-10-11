#%%
from graph_builder import GraphBuilder
from graph_renderer import GraphVisualizer

#%%
# a = GraphBuilder()
# a.buildGraphLogistic("data/tbi_elix_btlabels.xlsx")
# a.exportGraph("data/test_comorb.xml")

#%%
b = GraphVisualizer("data/test_comorb.xml")
b.genRenderArgs()
b.genLegend()
# b.renderGraphNX(cmap=False, adjust_shell=False)
b.renderScatter()

# %%
