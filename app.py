import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output
import numpy as np

# Read stakeholder data from Excel
df = pd.read_excel('stakeholders.xlsx', header=5)

# Create a graph
G = nx.Graph()

# Function to determine node color based on 'Power of influence'


def get_node_color(power):
    if power == 'A':
        return '#4CAF50'  # Green
    elif power == 'B':
        return '#FFA500'  # Orange
    else:
        return '#9E9E9E'  # Gray

# Function to determine node size based on 'Interest'


def get_node_size(interest):
    if interest == 'A':
        return 30
    elif interest == 'B':
        return 20
    else:
        return 15


def create_legend():
    return html.Div([
        html.H5("Legend"),
        html.Div([
            html.Span("■ ", style={'color': '#4CAF50', 'font-size': '20px'}),
            "High stakeholder influence: Significant ability to impact project decisions and outcomes."
        ]),
        html.Div([
            html.Span("■ ", style={'color': '#FFA500', 'font-size': '20px'}),
            "Medium stakeholder influence: Some influence, but not critical to decision-making."
        ]),
        html.Div([
            html.Span("■ ", style={'color': '#9E9E9E', 'font-size': '20px'}),
            "Low stakeholder influence: Little to no influence over the project."
        ]),
        html.Div([
            html.Span("■ ", style={'color': '#8A2BE2', 'font-size': '20px'}),
            "Project partner"
        ]),
    ], style={'margin-top': '20px'})

# Add project partner nodes with initial positions


project_partners = df['Project partner'].unique()
initial_pos = {}
for i, partner in enumerate(project_partners):
    angle = 2 * np.pi * i / len(project_partners)
    x = np.cos(angle) * 7
    y = np.sin(angle) * 7
    G.add_node(partner, size=100, color='#8A2BE2', type='partner', label=partner)
    initial_pos[partner] = (x, y)

# Add stakeholder nodes and edges
for _, row in df.iterrows():
    stakeholder_name = row['Stakeholder name']
    project_partner = row['Project partner']
    power = row['Power of influence']
    interest = row['Interest']

    G.add_node(stakeholder_name,
               size=get_node_size(interest),
               color=get_node_color(power),
               type='stakeholder',
               description=row['Stakeholder description'],
               stakeholder_type=row['Stakeholder type'],
               country=row['Country NUTS I'],
               region=row['Region NUTS II'],
               primary_category=row['Primary category'],
               secondary_category=row['Secondary category'],
               interest=interest,
               power=power,
               co_design=row['Co-design process'],
               mainstreaming=row['Mainstreaming and communication'],
               living_labs=row['Living labs'],
               training=row['Training'],
               pilot=row['Pilot city'],
               idss=row['IDSS'])
    G.add_edge(project_partner, stakeholder_name)

# Create a layout for the graph
pos = nx.spring_layout(G, k=1.6, iterations=50, pos=initial_pos, fixed=project_partners)

# Get unique, non-null stakeholder types
stakeholder_types = df['Stakeholder type'].dropna().unique()

# Get unique, non-null primary categories
primary_categories = df['Primary category'].dropna().unique()

# Create Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("", className="text-center mb-4"), width=12)
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Filters", className="card-title"),
                    dbc.Tooltip("Select stakeholder type(s)", target="stakeholder-type-filter", placement="right"),
                    dcc.Dropdown(
                        id='stakeholder-type-filter',
                        options=[{'label': t, 'value': t} for t in stakeholder_types if pd.notnull(t)],
                        multi=True,
                        placeholder="Select stakeholder type(s)"
                    ),
                    html.Br(),
                    dbc.Tooltip("Select country(ies)", target="country-filter", placement="right"),
                    dcc.Dropdown(
                        id='country-filter',
                        options=[{'label': c, 'value': c} for c in df['Country NUTS I'].dropna().unique()],
                        multi=True,
                        placeholder="Select country(ies)"
                    ),
                    html.Br(),
                    dbc.Tooltip("Select segment(s) of stakeholder cooperation", target="cooperation-filter", placement="right"),
                    dcc.Dropdown(
                        id='cooperation-filter',
                        options=[
                            {'label': 'Co-design process', 'value': 'co_design'},
                            {'label': 'Mainstreaming and communication', 'value': 'mainstreaming'},
                            {'label': 'Living labs', 'value': 'living_labs'},
                            {'label': 'Training', 'value': 'training'},
                            {'label': 'Pilot city', 'value': 'pilot_city'},
                            {'label': 'IDSS', 'value': 'idss'}
                        ],
                        multi=True,
                        placeholder="Select segment(s) of stakeholder cooperation"
                    ),
                    html.Br(),
                    dbc.Tooltip("Select stakeholder influence level(s)", target="power-filter", placement="right"),
                    dcc.Dropdown(
                        id='power-filter',
                        options=[
                            {'label': 'High stakeholder influence', 'value': 'A'},
                            {'label': 'Medium stakeholder influence', 'value': 'B'},
                            {'label': 'Low stakeholder influence', 'value': 'C'}
                        ],
                        multi=True,
                        placeholder="Select stakeholder influence level(s)"
                    ),
                    html.Br(),
                    dbc.Tooltip("Select primary stakeholder categories", target="primary-category-filter", placement="right"),
                    dcc.Dropdown(
                        id='primary-category-filter',
                        options=[{'label': c, 'value': c} for c in primary_categories if pd.notnull(c)],
                        multi=True,
                        placeholder="Select primary stakeholder category(ies)"
                    ),
                ])
            ], className="mb-4"),
            create_legend()  # Add the legend here
        ], width=3),
        dbc.Col([
            dcc.Loading(
                id="loading",
                type="default",
                children=[dcc.Graph(id='stakeholder-graph', style={'height': '80vh'})]
            )
        ], width=9)
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='node-info', className="mt-4"))
    ])
], fluid=True)


@app.callback(
    Output('stakeholder-graph', 'figure'),
    [Input('stakeholder-type-filter', 'value'),
     Input('country-filter', 'value'),
     Input('cooperation-filter', 'value'),
     Input('power-filter', 'value'),
     Input('primary-category-filter', 'value')]
)
def update_graph(stakeholder_types, countries, cooperation, power_levels, primary_categories):
    # Always include project partners
    filtered_nodes = set(node for node in G.nodes() if G.nodes[node]['type'] == 'partner')

    stakeholder_nodes = set(node for node in G.nodes() if G.nodes[node]['type'] == 'stakeholder')

    if stakeholder_types:
        stakeholder_nodes = {node for node in stakeholder_nodes if G.nodes[node].get('stakeholder_type') in stakeholder_types}
    if countries:
        stakeholder_nodes = {node for node in stakeholder_nodes if G.nodes[node].get('country') in countries}
    if cooperation:
        stakeholder_nodes = {node for node in stakeholder_nodes if any(G.nodes[node].get(coop) == 'Yes' for coop in cooperation)}
    if power_levels:
        stakeholder_nodes = {node for node in stakeholder_nodes if G.nodes[node].get('power') in power_levels}
    if primary_categories:
        stakeholder_nodes = {node for node in stakeholder_nodes if G.nodes[node].get('primary_category') in primary_categories}

    filtered_nodes.update(stakeholder_nodes)

    subgraph = G.subgraph(filtered_nodes)
    pos_subgraph = {node: pos[node] for node in subgraph.nodes()}

    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    for edge in subgraph.edges():
        x0, y0 = pos_subgraph[edge[0]]
        x1, y1 = pos_subgraph[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    node_trace = go.Scatter(
        x=[], y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=[],
            size=[],
            opacity=1,
            line_width=5))

    annotations = []

    for node in subgraph.nodes():
        x, y = pos_subgraph[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_info = subgraph.nodes[node]
        node_trace['marker']['color'] += (node_info['color'],)
        node_trace['marker']['size'] += (node_info['size'],)
        node_info_text = f"<b>{node}</b><br>"
        if node_info['type'] == 'stakeholder':
            node_info_text += f"Type: {node_info['stakeholder_type']}<br>"
            node_info_text += f"Country: {node_info['country']}<br>"
            node_info_text += f"Power: {node_info['power']}<br>"
            node_info_text += f"Interest: {node_info['interest']}<br>"
            node_info_text += f"Primary Category: {node_info['primary_category']}<br>"
        node_trace['text'] += (node_info_text,)

        # Add annotation for project partner nodes
        if node_info['type'] == 'partner':
            annotations.append(
                dict(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text=node,
                    showarrow=False,
                    font=dict(size=10, color="#ffffff"),
                )
            )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        clickmode='event+select',
                        annotations=annotations  # Add annotations to the layout
                    ))

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial, sans-serif", size=12, color="#333333"),
        dragmode='pan',  # Enable panning
        newshape=dict(line_color='#000'),
        height=800
    )

    return fig


@app.callback(
    Output('node-info', 'children'),
    [Input('stakeholder-graph', 'clickData')]
)
def display_click_data(clickData):
    if clickData is None:
        return html.P("Click on a stakeholder to see more information")
    else:
        node_name = clickData['points'][0]['text'].split('<br>')[0].replace('<b>', '').replace('</b>', '')
        node_info = G.nodes[node_name]

        info_list = [
            html.Li(f"{key.capitalize()}: {value}")
            for key, value in node_info.items()
            if key != 'color' and key != 'size'
        ]

        return html.Div([
            html.H4("Stakeholder information"),
            html.Ul(info_list)
        ])

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
