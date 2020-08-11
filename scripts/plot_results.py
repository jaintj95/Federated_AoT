import plotly.graph_objects as go
import plotly.express as pxl
import csv
import pandas as pd

def plot_data(filename, defence_type=''):
    # Create random data with numpy
    df = pd.read_csv(filename)
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['fl_iter'], y=df['main_task_acc'],
                        mode='lines',
                        name='test'))
    fig.add_trace(go.Scatter(x=df['fl_iter'], y=df['backdoor_acc'],
                        mode='lines', name='target'))
    x_title = '# Rounds'
    y_title = 'Success Probability'
    title = 'Defence Type: ' + str(defence_type) 
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.show()