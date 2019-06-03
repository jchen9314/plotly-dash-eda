# -*- coding: utf-8 -*-
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import flask
import plotly.plotly as py
import matplotlib.pyplot as plt
import plotly.tools as tls
import math
import os
from apps import coloranalyzer,camera_effect
import collections
import yaml
import warnings
# import argparse
warnings.filterwarnings("ignore")
plt.style.use("tableau-colorblind10")

server = flask.Flask(__name__)
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, server=server,
                external_stylesheets=external_stylesheets)
app.config.suppress_callback_exceptions = True
# parser = argparse.ArgumentParser()
# parser.add_argument("--input_csv_path", type=str,
#                     help="input EDA dataset folder path")
# parser.add_argument("--output_csv_path", type=str,
#                     help="output EDA dataset file path")
# args = parser.parse_args()

tab_style = {
    'padding': '10px',
    'fontWeight': 'bold',
    'marginBottom':'20px',
    'fontSize':18,
    'height':'50px'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'backgroundColor': 'rgb(0,87,168)',
    'padding': '10px',
    'color': 'white',
    'marginBottom':"20px",
    'height':'50px',
    'fontSize':20
}

app.layout = html.Div(
    [
        # header
        html.Div([
            html.Span("Colour Analyzer", className='app-title',
                    style={'color': 'white', 'fontSize': 38,
                           "marginLeft":"10px",'text-align':'center',
                           'marginTop':'20px','marginBottom':'20px'}),
            html.Img(src=app.get_asset_url('QIO-IRONN_logo-white-on-blue.png'),
                    style={'float':'right','height' : '8%','width' : '8%'})
            ],
            style={'marginTop': "20px",
                   'marginRight': "20px",
                   'marginLeft': "20px",
                   'backgroundColor': 'rgb(0,87,168)'}
        ),

        # tabs
        html.Div([

            dcc.Tabs(
                id="tabs",
                children=[
                    dcc.Tab(label="Color Analysis", value="coloranalyzer_tab",
                            style=tab_style,selected_style=tab_selected_style),
                    dcc.Tab(label="Camera Effect", value="camera_tab",
                            style=tab_style,selected_style=tab_selected_style)
                ],
                value="coloranalyzer_tab",
                style={"marginLeft":'20px',
                       'fontWeight':'bold',
                       'marginRight':'20px'}),
            ]),

        # Tab content
        html.Div(id="tab_content", className="row",
                 style={"margin": "5px 5px 5px 10px"}),
    ],
    className="row",
    style={"marginRight": "10px", "marginLeft":"10px"},
)

# def round_num(string):
#     if string == np.nan:
#         return string
#     else:
#         return round(string,2)


def create_dataset(file):
    df = pd.read_csv(file,index_col=0)
    df = df.dropna(subset=list(df.columns[2:11]), axis=0)
    df = df.fillna("N/A")
    for col in df.columns:
        if df[col].dtype != object:
            df[col] = df[col].apply(lambda x:round(x,2))
        elif col in ["total_zoom","exposure_time_sec"]:
            df[col] = [round(float(x),2) if x != "N/A" else "N/A" for x in df[col]]
        else:
            df[col] = df[col]
    return df

file = os.path.join("data","complete_joined_df.csv")

df = create_dataset(file)
dct = dict(collections.Counter(df.Type))
ls_type = []
for rock in dct.keys():
    if dct[rock] > 10:
        ls_type.append(rock)
all_options = {
    'CombinedType': ['CW', 'ORE', "DW"],
    'Type': ls_type}
camera_feat =  ["Focal length","Lens aperture","Zoom-in degree",
                "Megapixels", "Width/Height ratio"]
show_cols = ['file_name', 'Type', 'CombinedType',
             'SkewnessBlue', 'KurtosisBlue', 'MeanPixelBlue',
             'SkewnessGreen', 'KurtosisGreen', 'MeanPixelGreen',
             'SkewnessRed', 'KurtosisRed', 'MeanPixelRed',
             'camera','focal','lens','total_zoom','exposure_time_sec',
             'number_of_megapixels','Width_to_Height_Ratio']

def df_to_table(df):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [
            html.Tr(
                [
                    html.Td(df.iloc[i][col])
                    for col in df.columns
                ]
            )
            for i in range(len(df))
        ]
    )

# pure color analysis tab
@app.callback(
    Output('rocktypes_pc', 'options'),
    [Input('tier_pc', 'value')])
def set_rocktype_options(tier):
    return [{'label': i, 'value': i} for i in all_options[tier]]

@app.callback(
    Output('rocktypes_pc', 'value'),
    [Input('tier_pc', 'value')])
def set_rocktype_values(tier):
    if tier == "CombinedType":
        return ["ORE"]
    else:
        return ["HEM"]

@app.callback(
    Output('kdeplot_rock_pc', 'figure'),
    [Input('button-run-pc', 'n_clicks')],
    state=[State('feat_pc', 'value'),
     State('channel_pc', 'value'),
     State('tier_pc', 'value'),
     State('rocktypes_pc','value')
     ])
def update_graph(run_click,feat, channel, tier, rocktypes):
    if run_click:
        try:
            df = create_dataset(file)
            colname = [col for col in df.columns if channel in col and feat in col][0]
            groups = df.groupby(tier).groups
            fig,ax = plt.subplots(figsize=(10,4.5))
            ls = [df[colname][groups[x]] for x in rocktypes]
            for i in range(len(ls)):
                ls[i].plot.kde(ax=ax,label=rocktypes[i],linewidth=3)
            plt.title(feat+" on "+channel+" Channel",fontsize=20)
            if feat == "Skewness":
                plt.xlim((min(df[colname]),max(df[colname])))
            else:
                plt.xlim((min(df[colname])-50,max(df[colname])+50))
            plt.tick_params(labelsize=15)
            plt.ylabel("")
            plotly_fig = tls.mpl_to_plotly(fig)
            plotly_fig["layout"]["showlegend"]=True
            plotly_fig["layout"]["autosize"]=True
            plotly_fig["layout"]["hovermode"]="x"
            plotly_fig["layout"]["legend"]={"font":{"size":"18"}}
            print(type(plotly_fig))
            return plotly_fig
        except:
            return {'data':[], 'layout':[]}

@app.callback(
    Output("img_table_pc", "children"),
    [Input('button-generate-pc',"n_clicks")],
    [State('feat_pc', 'value'),
    State('channel_pc', 'value'),
    State("feat_lower_bound_pc", "value"),
    State("feat_upper_bound_pc", "value"),
    State("rocktypes_pc","value"),
    State("tier_pc","value")]
)
def leads_table_callback(generate_button,feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rocks, tier):
    if generate_button:
        df = create_dataset(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        tmp = df[df[tier].isin(rocks)]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        return df_to_table(res)

@app.callback(
    Output('save-table-textbox-pc', 'children'),
    [Input('button-save-pc', 'n_clicks')],
    [State('feat_pc', 'value'),
    State('channel_pc', 'value'),
    State("feat_lower_bound_pc", "value"),
    State("feat_upper_bound_pc", "value"),
    State("rocktypes_pc","value"),
    State("tier_pc","value")]
)
def save_current_table(savebutton, feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rocks, tier):
    if savebutton:
        df = create_dataset(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        tmp = df[df[tier].isin(rocks)]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        res.to_csv(os.path.join("data","tmp_color_analysis_result.csv"))
        return "Table saved."

# camera effect tab
def create_tag_set(df, feat_name, feat_options, tag_name):
    dff = df[df[feat_name] == feat_options[0]]
    lst = [x for x in dff[tag_name].unique()]
    if len(feat_options) == 1:
        return dff, lst
    else:
        for name in feat_options[1:]:
            dff = pd.concat([dff, df[df[feat_name]==name]])
            lst = list(set(lst) & set([tag for tag in df[df[feat_name]==name][tag_name].unique()]))
        return dff,lst

@app.callback(
    Output('rock_ce', 'options'),
    [Input('tier_ce', 'value')])
def set_rock_options(tier):
    return [{'label': i, 'value': i} for i in all_options[tier]]

@app.callback(
    Output('rock_ce', 'value'),
    [Input('tier_ce', 'value')])
def set_rock_values(tier):
    if tier == "CombinedType":
        return "ORE"
    else:
        return "HEM"

@app.callback(
    Output("camera_model_ce","options"),
    [Input("tier_ce","value"),
    Input("rock_ce","value")])
def set_camera_options(tier, rock):
    df = create_dataset(file)
    dff = df[df[tier] == rock]
    dct = dict(collections.Counter(dff.camera))
    ls_camera = []
    for camera in dct.keys():
        if dct[camera] > 10:
            ls_camera.append(camera)
    return [{'label': i, 'value': i} for i in ls_camera]

@app.callback(
    Output("camera_model_ce","value"),
    [Input("tier_ce","value"),
    Input("rock_ce","value")])
def set_camera_value(tier, rock):
    df = create_dataset(file)
    dff = df[df[tier] == rock]
    return [dff["camera"].unique()[0]]

@app.callback(
    Output('tag_val_ce', 'options'),
    [Input('tier_ce', 'value'),
     Input('rock_ce', 'value'),
     Input('camera_model_ce', 'value'),
     Input('tag_name_ce', 'value')
     ])
def set_tag_options(tier, rock, camera_model,tag_name):
    df = create_dataset(file)
    df = df[df[tier] == rock]
    if tag_name == "Focal length":
        dff,res_lst = create_tag_set(df, "camera", camera_model, "focal_comb")
        return [{'label': i, 'value': i} for i in res_lst]
    elif tag_name == "Lens aperture":
        dff,res_lst = create_tag_set(df, "camera", camera_model, "lens_comb")
        return [{'label': i, 'value': i} for i in res_lst]
    elif tag_name == "Zoom-in degree":
        dff,res_lst = create_tag_set(df, "camera", camera_model, "zoom_comb")
        return [{'label': i, 'value': i} for i in res_lst]
    elif tag_name == "Megapixels":
        dff,res_lst = create_tag_set(df, "camera", camera_model, "megapx_comb")
        return [{'label': i, 'value': i} for i in res_lst]
    else:
        dff,res_lst = create_tag_set(df, "camera", camera_model, "ratio_comb")
        return [{'label': i, 'value': i} for i in res_lst]

@app.callback(
    Output('tag_val_ce', 'value'),
    [Input('tier_ce', 'value'),
     Input('rock_ce', 'value'),
     Input('camera_model_ce', 'value'),
     Input('tag_name_ce', 'value')
     ])
def set_tag_values(tier, rock, camera_model, tag_name):
    df = create_dataset(file)
    df = df[df[tier] == rock]
    if tag_name == "Focal length":
        dff, lst = create_tag_set(df, "camera", camera_model, "focal_comb")
        if len(lst)>=1:
            return lst[0]
    elif tag_name == "Lens aperture":
        dff,lst = create_tag_set(df, "camera", camera_model, "lens_comb")
        if len(lst)>=1:
            return lst[0]
    elif tag_name == "Zoom-in degree":
        dff,lst = create_tag_set(df, "camera", camera_model, "zoom_comb")
        if len(lst)>=1:
            return lst[0]
    elif tag_name == "Megapixels":
        dff,lst = create_tag_set(df, "camera", camera_model, "megapx_comb")
        if len(lst)>=1:
            return lst[0]
    else:
        dff,lst = create_tag_set(df, "camera", camera_model, "ratio_comb")
        if len(lst)>=1:
            return lst[0]

@app.callback(
    Output('kdeplot_camera_ce', 'figure'),
    [Input('button-run-ce','n_clicks')],
    [State('feat_ce', 'value'),
     State('channel_ce', 'value'),
     State('tier_ce', 'value'),
     State('rock_ce', 'value'),
     State('camera_model_ce', 'value'), # multi-values
     State('tag_name_ce', 'value'),
     State('tag_val_ce', 'value')
     ])
def update_graph_camera(run_clicks, feat, channel, tier, rock, camera_model, tag_name, tag_val):
    if run_clicks:
        try:
            df = create_dataset(file)
            colname = [col for col in df.columns if channel in col and feat in col][0]
            df = df[df[tier] == rock]
            if tag_val:
            # camera_groups = df.groupby("camera_comb").groups
                fig,ax = plt.subplots(figsize=(10,6))
                if tag_name == "Focal length":
                    dff,res_lst = create_tag_set(df, "camera", camera_model, "focal_comb")
                    ls = [dff[(dff["camera"]==x) &\
                              (dff[tier]==rock) &\
                              (dff["focal_comb"]==tag_val)][colname] for x in camera_model]
                elif tag_name == "Lens aperture":
                    dff,res_lst = create_tag_set(df, "camera", camera_model, "lens_comb")
                    ls = [dff[(dff["camera"]==x) &\
                              (dff[tier]==rock) &\
                              (dff["lens_comb"]==tag_val)][colname] for x in camera_model]
                elif tag_name == "Zoom-in degree":
                    dff,res_lst = create_tag_set(df, "camera", camera_model, "zoom_comb")
                    ls = [dff[(dff["camera"]==x) &\
                              (dff[tier]==rock) &\
                              (dff["zoom_comb"]==tag_val)][colname] for x in camera_model]
                elif tag_name == "Megapixels":
                    dff,res_lst = create_tag_set(df, "camera", camera_model, "megapx_comb")
                    ls = [dff[(dff["camera"]==x) &\
                              (dff[tier]==rock) &\
                              (dff["megapx_comb"]==tag_val)][colname] for x in camera_model]
                else:
                    dff,res_lst = create_tag_set(df,"camera", camera_model, "ratio_comb")
                    ls = [dff[(dff["camera"]==x) &\
                              (dff[tier]==rock) &\
                              (dff["ratio_comb"]==tag_val)][colname] for x in camera_model]
                for i in range(len(ls)):
                    ls[i].plot.kde(ax=ax,label=camera_model[i],linewidth=3)
                plt.title("Camera Effect"+" on "+rock + " with Filtered " + tag_name,fontsize=20)
                if feat == "Skewness":
                    plt.xlim((min(dff[colname]-1),max(dff[colname])+1))
                else:
                    plt.xlim((min(dff[colname])-50,max(dff[colname])+50))
                plt.tick_params(labelsize=15)
                plt.ylabel("")
                plotly_fig = tls.mpl_to_plotly(fig)
                plotly_fig["layout"]["showlegend"]=True
                plotly_fig["layout"]["autosize"]=True
                plotly_fig["layout"]["hovermode"]="x"
                plotly_fig["layout"]["legend"]={"font":{"size":"18"}}
                return plotly_fig
            else:
                fig,ax = plt.subplots(figsize=(10,6))
                ls = [df[(df["camera"]==x) &\
                          (df[tier]==rock)][colname] for x in camera_model]
                for i in range(len(ls)):
                    ls[i].plot.kde(ax=ax,label=camera_model[i],linewidth=3)
                plt.title("Camera Effect"+" on "+rock + " without Filters",fontsize=20)

                    # plt.title(feat+" on "+channel+" Channel",fontsize=20)
                if feat == "Skewness":
                    plt.xlim((min(df[colname])-1,max(df[colname])+1))
                else:
                    plt.xlim((min(df[colname])-50,max(df[colname])+50))
                plt.tick_params(labelsize=15)
                plt.ylabel("")
                plotly_fig = tls.mpl_to_plotly(fig)
                plotly_fig["layout"]["showlegend"]=True
                plotly_fig["layout"]["autosize"]=True
                plotly_fig["layout"]["hovermode"]="x"
                plotly_fig["layout"]["legend"]={"font":{"size":"18"}}
                return plotly_fig
        except:
            return {'data':[], 'layout':[]}

@app.callback(
    Output("img_table_ce", "children"),
    [Input('button-generate-ce',"n_clicks")],
    [State('feat_ce', 'value'),
    State('channel_ce', 'value'),
    State("feat_lower_bound_ce", "value"),
    State("feat_upper_bound_ce", "value"),
    State("rock_ce","value"),
    State("tier_ce","value"),
    State("camera_model_ce","value")]
)
def leads_table_callback(generate_button,feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rock, tier,model):
    if generate_button:
        df = create_dataset(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        tmp = df[(df[tier] == rock) & (df["camera"].isin(model))]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        return df_to_table(res)

@app.callback(
    Output('save-table-textbox-ce', 'children'),
    [Input('button-save-ce', 'n_clicks')],
    [State('feat_ce', 'value'),
    State('channel_ce', 'value'),
    State("feat_lower_bound_ce", "value"),
    State("feat_upper_bound_ce", "value"),
    State("rock_ce","value"),
    State("tier_ce","value"),
    State("camera_model_ce","value")]
)
def save_current_table(savebutton, feat_name, channel, feat_lower_bound,
                        feat_upper_bound, rock, tier, model):
    if savebutton:
        df = create_dataset(file)
        colname = [col for col in df.columns if channel in col and feat_name in col][0]
        tmp = df[(df[tier] == rock) & (df["camera"].isin(model))]
        res = tmp[(tmp[colname]>=yaml.load(feat_lower_bound)) &\
                  (tmp[colname]<=yaml.load(feat_upper_bound))][show_cols]
        res.to_csv(os.path.join("data","tmp_camera_effect_result.csv"))
        return "Table saved."


@app.callback(
    Output("tab_content", "children"),
    [Input("tabs", "value")]
    )
def render_content(tab):
    if tab == "coloranalyzer_tab":
        return coloranalyzer.app.layout
    else:
        return camera_effect.app.layout


if __name__ == "__main__":
    app.run_server(debug=True)
