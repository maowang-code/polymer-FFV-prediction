# -*- coding: utf-8 -*-
"""
interactive graph of the FFV prediction
Author: chewmao@nus.edu.sg
"""



import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash 
from dash import dcc, html, Input, Output

import dash_dangerously_set_inner_html as dhtml
import dash_bootstrap_components as dbc


from rdkit import Chem 
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdDepictor

import joblib   # sklearn version 0.232





def ffv_plot(df, x_axis_column_name, y_axis_column_name, colour_column_value):
    """
    Plotly Figure object for ffv data

    Parameters
    ----------
    df: DataFrame
    x_axis_column_name: str
    y_axis_column_name: str
    colour_column_value: str

    Returns
    -------
    Figure
        Plotly figure object
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        # X and Y coordinates from data table
        x=df[x_axis_column_name],
        y=df[y_axis_column_name],
        text=df.index,
        mode='markers',
        #hover_name="country",
        # Set the format of scatter
        marker=dict(
            symbol='circle',
            opacity=1,
            line=dict(color='rgb(40, 40, 40)', width=0.2),
            size=4,
            # Colour bar
            color=df[colour_column_value],
            colorscale='jet',
            colorbar=dict(
                thicknessmode='pixels',
                thickness=15,
                title=dict(text=colour_column_value, side='right')
            ),
            reversescale=False,
            showscale=True
        )
    ))
    # Set the format of axes
    axis_template = dict(linecolor='#444', tickcolor='#444',
                         ticks='outside', showline=True, zeroline=False,
                         gridcolor='lightgray')
    fig.update_layout(
        xaxis=dict(axis_template, **dict(title=x_axis_column_name)),
        yaxis=dict(axis_template, **dict(title=y_axis_column_name)),
        clickmode='event+select',
        hovermode='closest',
        plot_bgcolor='white'
    )
    return fig


def smi2svg(smi):
    """
    draw polymer structure

    Parameters
    ----------
    smi: SMILES string of polymer 


    Returns
    -------
    SVG figure
        
    """
    mol = Chem.MolFromSmiles(smi)
    rdDepictor.Compute2DCoords(mol)
    mc = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mc)
    drawer = Draw.MolDraw2DSVG(300,300)
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:','')
    return svg



def fragmentfp(smiles):
    """
    generate 255-bit fragment fingerprints

    Parameters
    ----------
    smiles: SMILES string of polymer 


    Returns
    -------
    an 255-bit fingerprint (list)
        
    """
    frag255=pd.read_csv('./data/frag255.csv')
    fragemtsall = list(frag255['frag_smiles'])
    fp=[[] for _ in range(255)] 
    mol = Chem.MolFromSmiles(smiles)
    if mol==None:
        #print('error')
        return None
    
    else:
        for j in range(len(fragemtsall)):       
            core = Chem.MolFromSmarts( fragemtsall[j] )
            has_sub = list(mol.GetSubstructMatches(core))     ## GetSubstructMatches will list all the substructure
            subnb = len(has_sub)
            fp[j]=subnb
    return fp



def ML_prediction(fp):
    """
    predict polymer density,van der Waals volume, FFV

    Parameters
    ----------
    fp: 255-bit fingerprint


    Returns
    -------
    ML predicted density,van der Waals volume, FFV
        
    """
    density_model = joblib.load('./data/fragment255_density_100_rf1_0.sav')
    volume_model = joblib.load('./data/fragment255_volume_100_rf1_0.sav')
    scaler = joblib.load('./data/scaler.joblib')
    
    density_pred =density_model.predict(scaler.transform([fp]))
    volume_pred =volume_model.predict(scaler.transform([fp]))
    FFV_pred = 1-1.3*density_pred*volume_pred
    return density_pred[0],volume_pred[0],FFV_pred[0]







# Load dataframe
# random.seed(1234)
# slice=random.sample(range(29482),3000)
df = pd.read_csv('./data/29482_screened_polymer.csv').iloc[:3000,:]        
df.index=range(len(df))

columns_dict = [{'label': i, 'value': i} for i in df.columns[1:]]






###------1st panel setting------------
controls1 = dbc.Card(
    [   html.H4("Choosing the plot axies", className="card-title"),
        
       html.Hr(),

     
        dbc.FormGroup(
            [
                dbc.Label("Dataset"),
                dcc.RadioItems(
                    id='Dataset',
                    options=[
                        {'label': ' PI1M (predicted FFV > 0.2)', 'value': 'PI1M'}
                    ],
                    value='PI1M'
                ),
            ]
        ),
        

        
        
        dbc.FormGroup(
            [
                dbc.Label("X variable"),
                dcc.Dropdown(
                            id='x_axis_column',
                            className='axis_controller',
                            options=columns_dict,
                            value='Density (g/cm^3)'
                        ),
            ]
        ),
        
                
        dbc.FormGroup(
            [
                dbc.Label("Y variable"),
                dcc.Dropdown(
                            id='y_axis_column',
                            className='axis_controller',
                            options=columns_dict,
                            value='van der Waals volume (cm^3/g)'
                        ),
            ]
        ),
        
        
        dbc.FormGroup(
            [
                dbc.Label("Colorbar"),
                dcc.Dropdown(
                            id='colour_column',
                            options=columns_dict,
                            value='FFV'
                        ),
            ]
        ),
        
        
        dbc.FormGroup(
            [
                dbc.Label("Range slider"),
                dcc.Dropdown(
                            id='range_column',
                            options=columns_dict,
                            value='FFV'
                        ),
                html.Br(),
                dcc.RangeSlider(min=0, max=2, step=1, value=[0.5,1],marks=None, id='range-slider'),
                html.P(id='selected_data'),
                
            ]
        ),        
        
        html.H6(
            dbc.Label(["*SCScore is to evaluate the synthesizability of a molecule (the lower, the easier to synthesize). Details can be found",
                  html.A(' HERE',href='http://dx.doi.org/10.1021/acs.jcim.7b00622',target='_blank'),'.']),
                ),  

        html.H6(
        dbc.Label(["*Information about PI1M dataset can be found",
                  html.A(' HERE',href='https://dx.doi.org/10.1021/acs.jcim.0c00726',target='_blank'),'.']),

                )
    ],
    body=True,
    outline=False,  # True = remove the block colors from the background and header
    color="light",
    style={"width": "18rem",'height':750},
)

###---2nd panel---graph-----------
controls2 = dbc.Card(
    [   html.H4("Data visulization", className="card-title"),
        #html.Hr(),

        # dbc.Alert(
        #     [
        #         html.I(className="bi2"),
        #         "Data visulization",
        #     ],
        #     color="primary",
            
        #     className="c2",
        # ), 
             
         
         
        dbc.FormGroup(
            [
                #dbc.Label("Graph"),
                dcc.Graph(id='indicator-graphic',style={'height':600}),
            ]
        ),
       
        # html.H6("Only 3000 points are shown here to reduce the server load.", className="card-end1"),
        # html.H6("For the entire data, please refer to related work.", className="card-end2"),
        
        # dbc.Label(["Only 5000 points are shown here to reduce the server load. For the entire data, please refer to related",
        # html.A(' work',href='https://www.researchgate.net/profile/Mao-Wang-16',target='_blank'),'.']),
        
        dbc.Label(["Only 3000 points are shown for clarity. To download 29482 shortlisted polymers:"]), 
        html.Button("Download 29482 shortlisted polymers", id="btn_excel"),
        dcc.Download(id="download-dataframe-excel"),    
        
        
    ],
    body=True,
    #color="light",   # https://bootswatch.com/default/ for more card colors
    #inverse=True,   # change color of text (black or white)
    outline=False,  # True = remove the block colors from the background and header
    style={"width": "36rem",'height':750},
)



###-----3rd panel-----------
controls3 = dbc.Card(
    [   html.H4(children="Selected Polymer Info", className="card-title"),
        #html.Hr(),

     # dbc.Alert(
     #        [
     #            html.I(className="bi3"),
     #            "Selected Polymer Info",
     #        ],
     #        color="primary",
            
     #        className="c3",
     #    ), 
     
     
        dbc.FormGroup(
            [
                #dbc.Label("[Please select a data point in the plot]"),
                
                html.Div(id="show_structure" ,className='structure',style={'textAlign': 'center'}),
                
            ]
        ),
        
        
        dbc.FormGroup(
            [
                #dbc.Label("Polymer SMILES (connecting points are represented by * ):"),
                html.H5([dbc.Badge("Polymer SMILES (connecting points are represented by *):",color="secondary")]),
                html.P(id='loading_selected_smiles' ,className='structure_smiles'),
            
                #dbc.Label("X variable value:"),
                html.H5([dbc.Badge("X variable value:",color="secondary")]),
                html.P(id='loading_selected_x' ,className='point_x'),
            
                #dbc.Label("Y variable value:"),
                html.H5([dbc.Badge("Y variable value:",color="secondary")]),
                html.P(id='loading_selected_y' ,className='point_y'),
    
                #dbc.Label("Colorbar value:"),
                html.H5([dbc.Badge("Colorbar value:",color="secondary")]),
                html.P(id='loading_selected_z' ,className='point_z'),
            ]
        ),        
        
    ],
    body=True,
    #color="light",   # https://bootswatch.com/default/ for more card colors
    #inverse=True,   # change color of text (black or white)
    outline=False,  # True = remove the block colors from the background and header
    style={"width": "27rem",'height':750},
)



###-----predict-- polymer panel-----------
controls4 = dbc.Card(
    [   #html.H4("FFV prediction", className="card-title"),
     
          html.H4(dbc.Alert(
            [ "FFV prediction"],
             color="primary",
            #  className="c4",
            ), ),    
     
        dbc.FormGroup(
            [   
             
                # dbc.Label("Please input the polymer repeat unit in"),
                # html.A(' SMILES ', href='https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system',target="_blank"),
                # #dbc.CardLink(" SMILES ", href="https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system", target="_blank"),
                # dbc.Label(" form or you can"),
                # html.A(" draw", href="https://pubchem.ncbi.nlm.nih.gov//edit3/index.html", target="_blank"),
                # dbc.Label(" the repeat unit to generate SMILES (connecting points are represented by * )"),
                
                
                dbc.Label(['Please keyin the repeat unit of a polymer in', html.A(' SMILES', href='https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system',target="_blank"),
                            ' or', html.A(' DRAW', href='https://pubchem.ncbi.nlm.nih.gov//edit3/index.html',target="_blank"),' the structure to generate SMILES (connecting points are represented by * )']),
            
                dbc.Input(id="input-on-submit", value="*Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)*"),
                

                
            ]
        ),

        
     
        dbc.FormGroup(
            [
                #dbc.Label("Polymer structure:"),
                html.H5([dbc.Badge("Polymer structure:",color="secondary")]),
                
                html.Div(id="show_structure_pre" ,className='structure_pre',style={'textAlign': 'center'}),
                
            ]
        ),
        
                
        
        dbc.FormGroup(
            [
                #dbc.Label("Predicted density (g/cm^3):"),
                html.H5([dbc.Badge("Predicted density (g/cm^3):",color="secondary")]),
                
                html.P(id='pre_density' ,className='pre_x'),
 
                #dbc.Label("Predicted van der Waals volume (cm^3/g):"),
                html.H5([dbc.Badge("Predicted van der Waals volume (cm^3/g):",color="secondary")]),
                html.P(id='pre_volume' ,className='pre_y'),

                #dbc.Label("Predicted FFV:"),
                html.H5([dbc.Badge("Predicted FFV:",color="secondary")]),

                html.P(id='pre_ffv' ,className='pre_z'),
            ]
        ), 
        
        
        
    ],
    body=True,
    #color="light",   # https://bootswatch.com/default/ for more card colors
    #inverse=True,   # change color of text (black or white)
    outline=False,  # True = remove the block colors from the background and header
    style={"width": "30rem",'height':750},
)




# Set up web server
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])   ##YETI,
server = app.server


app.layout = dbc.Container(
    [   html.Br(),
        html.H1(children = "Accelerating Discovery of High Fractional Free Volume Polymers from a Data-Driven Approach",
                style={
                    #'color':
                    # 'font_family': "Times New Roman",
                    }

                ),
        


        html.H5(
            
            dbc.Label([html.A('Mao Wang',href='https://maowang-code.github.io/',target='_blank'),' and',
                  html.A(' Jianwen Jiang',href='https://cheed.nus.edu.sg/stf/chejj/',target='_blank')]
                      )
                  
                ),


        html.H6(children="Department of Chemical and Biomolecular Engineering, National University of Singapore, 117576, Singapore",
                style={
                    #'color':
                    # 'font_family': "Times New Roman",
                    }

                ),


        #html.H6("For a better experience, please open in full screen."),
        html.Hr(),
        dbc.Row(
            [    ### total width is not larger than 12
                dbc.Col(controls1, width="auto"),   ##use auto 
                dbc.Col(controls2, width="auto"),
                dbc.Col(controls3, width="auto"),
                dbc.Col(controls4, width="auto"),
            ],
            align="around",style={'height':750}
        ),
    ],
    fluid=True,
)






####################################

# Setting range slider properties
@app.callback(
    dash.dependencies.Output('range-slider', 'min'),
    [dash.dependencies.Input('range_column', 'value')])
def select_bar1(range_column_value):
    return df[range_column_value].min()


@app.callback(
    dash.dependencies.Output('range-slider', 'max'),
    [dash.dependencies.Input('range_column', 'value')])
def select_bar2(range_column_value):
    return df[range_column_value].max()


@app.callback(
    dash.dependencies.Output('range-slider', 'value'),
    [dash.dependencies.Input('range_column', 'value')])
def select_bar3(range_column_value):
    return [df[range_column_value].min(), df[range_column_value].max()]


@app.callback(
    dash.dependencies.Output('range-slider', 'step'),
    [dash.dependencies.Input('range_column', 'value')]
)
def range_step(range_column_value):
    step = (df[range_column_value].max() - df[range_column_value].min())/100
    return step


# Print the selected range
@app.callback(
    dash.dependencies.Output('selected_data', 'children'),
    [dash.dependencies.Input('range-slider', 'value'),
     dash.dependencies.Input('range_column', 'value')])
def callback(range_slider_value, range_column_value):
    return 'Polymers with {} between {:>.3f} and {:>.3f} are selected.'.format(
        range_column_value, range_slider_value[0], range_slider_value[1]
    )




####################################
#####---------------polymer info-----------------


@app.callback(
    [Output('loading_selected_smiles','children'),Output('loading_selected_x', 'children'),
     Output('loading_selected_y', 'children'),Output('loading_selected_z', 'children')],
    [Input('indicator-graphic', 'clickData')])
def display_click_data(clickData):
    if clickData is None:
        #fake_return ='[Please select a data point in the plot]'
        point_smiles = ''#'*COC(=O)N[Si](C)(C)O[Si](C)(C)C[Si](C)(C)*'
        point_x = ''
        point_y = ''
        point_z = ''
        return '','','',''
    else:
        point_index = int(clickData['points'][0]['pointIndex'])
        point_smiles = str(df['SMILES'][point_index])
        point_x = float(clickData['points'][0]['x'])
        point_y = float(clickData['points'][0]['y'])
        point_z = float(clickData['points'][0]['marker.color'])
        
        return '{}'.format(point_smiles),'{:.3f}'.format(point_x),'{:.3f}'.format(point_y),'{:.3f}'.format(point_z)


####################################
#-------------Chemical structure 3D viewer----------

@app.callback(
    Output('show_structure', 'children'),
    Input('indicator-graphic', 'clickData'))
def update_img(clickData):
    
    if clickData is None:
        return '[Please select a data point in the plot]' 
    
    else:
        point_index = int(clickData['points'][0]['pointIndex'])
        point_smiles = str(df['SMILES'][point_index])
    
        try:
            svg = smi2svg(point_smiles)
        except:
            svg = '[Please select a data point in the plot]'
        return dhtml.DangerouslySetInnerHTML(svg)




# Figure updated by different dash components
@app.callback(
    dash.dependencies.Output('indicator-graphic', 'figure'),
    [dash.dependencies.Input('x_axis_column', 'value'),
     dash.dependencies.Input('y_axis_column', 'value'),
     dash.dependencies.Input('colour_column', 'value'),
     dash.dependencies.Input('range_column', 'value'),
     dash.dependencies.Input('range-slider', 'value')])
def update_graph(x_axis_column_name, y_axis_column_name, colour_column_value,
                 range_column_value, range_slider_value):
    filtered_df = pd.DataFrame(
        data=df[
            (df[range_column_value] >= range_slider_value[0]) &
            (df[range_column_value] <= range_slider_value[1])]
    )
    # General ESF map
    fig = ffv_plot(filtered_df, x_axis_column_name, y_axis_column_name,
                   colour_column_value)
    return fig






#######################################
## setting the input smiles --and the output svg smiles  and ML prediction-


@app.callback(
    Output('show_structure_pre', 'children'),
    Input('input-on-submit', 'value'))

def update_img2(value):
    
    if value is None:
        remark='Please input polymer SMILES!'
        return dhtml.DangerouslySetInnerHTML(remark)
    
    elif value.count('*')!=2 :
        remark='[Warning] The number of connecting points * should be 2!'
        return dhtml.DangerouslySetInnerHTML(remark)     
    
    else:
        pre_smiles = "{}".format(value)
    
        try:
            svg = smi2svg(pre_smiles)
        except:
            svg = '[Warning] Please check the SMILES!'
        return dhtml.DangerouslySetInnerHTML(svg)


  

@app.callback(
    [Output('pre_density', 'children'),Output('pre_volume', 'children'),Output('pre_ffv', 'children'),],

    [Input('input-on-submit', 'value')])
def display_pred1(value):
    fp=fragmentfp(value)
    a,b,c= ML_prediction(fp)    
    return '{:.3f}'.format(a),'{:.3f}'.format(b),'{:.3f}'.format(c)
        

###----download-files--------              
@app.callback(
    Output("download-dataframe-excel", "data"),
    Input("btn_excel", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_file(
        "./data/29482_shortlisted polymers_PI1M.xlsx"
    )

#################--------------------###############################
#################--------------------###############################


if __name__ == '__main__':
    app.run_server(debug=False, port=8058)
