import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.tools as tls
import math
from dash.dependencies import Input, Output, State
import yaml
import flask
import warnings
import collections
warnings.filterwarnings("ignore")

plt.style.use("tableau-colorblind10")

server = flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server,external_stylesheets=external_stylesheets)

camera_feat =  ["Focal length","Lens aperture","Zoom-in degree", "Megapixels", "Width/Height ratio"]
app.layout = html.Div([
    html.Div([
        html.P("Select a colour feature", style={'font-weight': 'bold', 'margin-bottom': '3px', "font-size":"23px"}),
        dcc.RadioItems(
            id='feat_ce',
            options=[{'label': i, 'value': i} for i in ["Skewness","Kurtosis","MeanPixel"]],
            value='Skewness',
            inputStyle={"height":"20px","width":"20px"},
            labelStyle={'display': 'inline-block',
            'margin-right': '5px',
            'font-size':"20px", "margin-bottom":"3px"}
        ),
        html.P("Select a colour channel", style={'font-weight': 'bold', 'margin-bottom': '3px', "margin-top":"3px", "font-size":"23px"}),
        dcc.RadioItems(
            id='channel_ce',
            options=[{'label': i, 'value': i} for i in ["Blue","Green","Red"]],
            value='Red',
            inputStyle={"height":"20px","width":"20px"},
            labelStyle={'display': 'inline-block','margin-right': '7px',
            "font-size":"20px", "margin-bottom":"3px"}
        ),
        html.P("Select a rock type tier", style={'font-weight': 'bold', 'margin-bottom': '3px',"margin-top":"3px", "font-size":"23px"}),
        dcc.RadioItems(
            id='tier_ce',
            options=[
                {'label': 'CombinedType', 'value': 'CombinedType'},
                {'label': 'Type', 'value': 'Type'}],
            value="Type",
            inputStyle={"height":"20px","width":"20px"},
            labelStyle={'display':"inline-block",'margin-right':'7px',
            "font-size":"20px", "margin-bottom":"3px" }
        ),
        # html.P("Select rock types", style={'font-weight': 'bold', 'margin-bottom': '3px',"margin-top":"3px", "font-size":"23px"}),
        # dcc.Dropdown(
        #     id="rocktypes",
        #     multi=True,
        #     style={"margin-top":"10px","margin-bottom":"4px", "width":"95%", "font-size":"20px", "padding":"0 0 5px 0"}
        # ),
        html.P("Select a rock type of interest", style={'font-weight': 'bold', 'margin-bottom': '3px',"margin-top":"3px", "font-size":"23px"}),
        dcc.Dropdown(
            id="rock_ce",
            style={"margin-top":"10px","margin-bottom":"4px", "width":"95%", "font-size":"20px", "padding":"0 0 5px 0"}
        ),
        html.P("Select a camera model", style={'font-weight': 'bold', 'margin-bottom': '4px',"font-size":"25px"}),
        dcc.Dropdown(
            id="camera_model_ce",
            multi=True,
            style={"margin-top":"10px","margin-bottom":"4px", "width":"95%", "font-size":"20px", "padding":"0 0 5px 0"}
        ),
        html.P("Select an exif tag",style={'font-weight': 'bold', 'margin-bottom': '4px',"font-size":"25px"}),
        dcc.Dropdown(
            id="tag_name_ce",
            options=[{'label': i, 'value': i} for i in camera_feat],
            value="Focal length",
            style={"margin-top":"10px","margin-bottom":"4px", "width":"95%", "font-size":"20px", "padding":"0 0 5px 0"}
        ),
        # html.Div(id='container-button-basic',
        #      children="check empty dataframe")
        html.P("Select an avaible tag value", style={'font-weight': 'bold', 'margin-bottom': '4px',"font-size":"25px"}),
        dcc.Dropdown(
            id="tag_val_ce",
            style={"margin-top":"10px","margin-bottom":"4px", "width":"95%", "font-size":"20px", "padding":"0 0 5px 0"}
        ),
        html.Button('Run Analysis',id='button-run-ce',style={'margin-right': '20px', 'margin-top': '5px'}),
        # html.Button('Reset', id='button-reset', style={'margin-top': '5px'})
    ],style={
    "borderTop":"thin lightgrey solid",
    "borderRadius":"5px",
    'borderBottom': 'thin lightgrey solid',
    'borderLeft': 'thin lightgrey solid',
    'borderRight': 'thin lightgrey solid',
    'padding': '10px 30px 20px 30px',
    "margin-left":"30px",
    "margin-top":"30px",
    "margin-bottom":"40px",
    "display": "inline-block",
    "width":"40%",
    "float":"left",
    # "backgroundColor":"rgba"+str(tuple(color1))
    "backgroundColor":"rgb(255,255,255)"
    }),

    html.Div([
        dcc.Graph(
            id='kdeplot_camera_ce',
        )
    ], style={'display': 'inline-block', 'padding': '0px 10px 0px 0px', 'float':'right',"width":"46%","margin-right":"150px","margin-bottom":'26px'}),

    html.Div([
        dcc.Input(
            id="feat_lower_bound_ce",
            placeholder="Enter a start value",
            type='text',
            value='',
            style={'width':'25%','marginLeft': "20px","font-size":"20px"}),
        dcc.Input(
            id="feat_upper_bound_ce",
            placeholder="Enter an end value",
            type='text',
            value='',
            style={'width':'25%','marginLeft': "20px","font-size":"20px"}),
        html.Button('Generate Table',id='button-generate-ce',style={'marginBottom':'15px','marginLeft': '20px'}),
        html.Button('Save Table',id='button-save-ce',style={'marginLeft': '20px'}),
        html.Div(id='save-table-textbox-ce')

        ],style={
            "borderTop":"thin lightgrey solid",
            "borderRadius":"5px",
            'borderBottom': 'thin lightgrey solid',
            'borderLeft': 'thin lightgrey solid',
            'borderRight': 'thin lightgrey solid',
            'padding': '30px 30px 20px 20px',
            "margin-left":"30px",
            "margin-top":"10px",
            'margin-bottom': "40px",
            "display": "inline-block",
            "width":"40%",
            "float":"left",
            # "backgroundColor":"rgba"+str(tuple(color1))
            "backgroundColor":"rgb(255,255,255)"
        }),
    html.Div(
        id="img_table_ce",
        className="row",
        style={
            "maxHeight": "350px",
            "overflowY": "scroll",
            "padding": "8",
            "font-size":"20px",
            "backgroundColor":"white",
            "border": "1px solid #C8D4E3",
            "borderRadius": "3px",
            "width":'90%',
            "float":"left",
            'marginBottom': "10px",
            'marginTop': "30px",
            'marginRight': "20px",
            'marginLeft': "25px",
        })
])
