import plotly.graph_objects as go
import plotly.express as pxl
import csv
import pandas as pd

def plot_data(filename):
    # Create random data with numpy
    df = pd.read_csv(filename)
    
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['fl_iter'], y=df['main_task_acc'],
                        mode='lines',
                        name='test'))
    fig.add_trace(go.Scatter(x=df['fl_iter'], y=df['backdoor_acc'],
                        mode='lines', name='target'))

    fig.show()