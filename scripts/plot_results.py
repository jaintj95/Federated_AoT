import plotly.graph_objects as go
import plotly.express as pxl
import csv
import pandas as pd
import os
import json

def plot_data(filename, file_dir='results', destination_dir='images', defence_type=''):
    # Create random data with numpy
    df = pd.read_csv(os.path.join(file_dir, filename))
    
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
    image_filename = os.path.join(destination_dir, filename[:filename.index('.csv')] + '.png')
    fig.write_image(image_filename)
    
    
def plot_aggregated_data(filename, file_dir='results', destination_dir='images', defence_type=''):
    # Create random data with numpy
    obj = json.loads(open(os.path.join(file_dir, filename), 'r').read())
    output_main = {}
    output_back = {}
    final_round = '249'
    for key in obj:
        obj[key] = json.loads(obj[key])
        output_main[key] = obj[key]['main_task_acc'][final_round]
        output_back[key] = obj[key]['backdoor_acc'][final_round]
    
    x_indices = list(output_main.keys())
    
    
    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_indices, y=list(output_main.values()),
                        mode='lines',
                        name='test'))
    fig.add_trace(go.Scatter(x=x_indices, y=list(output_back.values()),
                        mode='lines', name='target'))
    x_title = '% Compromised Nodes'
    y_title = 'Success Probability'
    title = 'Defence Type: ' + str(defence_type) 
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.show()
    image_filename = os.path.join(destination_dir, filename[:filename.index('.json')] + '.png')
    fig.write_image(image_filename)