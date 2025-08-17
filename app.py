from datetime import date
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
import warnings
warnings.filterwarnings('ignore')


def streamlit_config():

    # page configuration
    st.set_page_config(page_title='Rendal Property', layout="wide")

    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(0,0,0,0);
    }

    </style>
    """
    st.markdown(page_background_color, unsafe_allow_html=True)

    # title and position
    st.markdown(f'<h1 style="text-align: center;">Rental Property Price Prediction</h1>',
                unsafe_allow_html=True)
    add_vertical_space(1)


# custom style for submit button - color and width

def style_submit_button():

    st.markdown("""
                    <style>
                    div.stButton > button:first-child {
                                                        background-color: #367F89;
                                                        color: white;
                                                        width: 70%}
                    </style>
                """, unsafe_allow_html=True)


# custom style for prediction result text - color and position

def style_prediction():

    st.markdown(
        """
            <style>
            .center-text {
                text-align: center;
                color: #20CA0C
            }
            </style>
            """,
        unsafe_allow_html=True
    )



class plotly:

    def pie_chart(df, x, y, title, title_x=0.20):

        fig = px.pie(df, names=x, values=y, hole=0.5, title=title)

        fig.update_layout(title_x=title_x, title_font_size=22)

        fig.update_traces(text=df[y], textinfo='percent+value',
                          textposition='outside',
                          textfont=dict(color='white'),
                          outsidetextfont=dict(size=14))

        st.plotly_chart(fig, use_container_width=True)


    def vertical_bar_chart(df, x, y, text, color, title, title_x=0.35):

        fig = px.bar(df, x=x, y=y, labels={x: '', y: ''}, title=title)

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        fig.update_layout(title_x=title_x, title_font_size=22)

        df[y] = df[y].astype(float)
        text_position = ['inside' if val >= max(
            df[y]) * 0.90 else 'outside' for val in df[y]]

        fig.update_traces(marker_color=color,
                          text=df[text],
                          textposition=text_position,
                          texttemplate='%{y}',
                          textfont=dict(size=14),
                          insidetextfont=dict(color='white'),
                          textangle=0,
                          hovertemplate='%{x}<br>%{y}')

        st.plotly_chart(fig, use_container_width=True, height=100)


    def scatter_chart(df, x, y, size, title):

        fig = px.scatter(data_frame=df, x=x, y=y, size=size, color=y, 
                         labels={x: '', y: ''}, title=title)
        
        fig.update_layout(title_x=0.4, title_font_size=22)
        
        fig.update_traces(hovertemplate=f"{x} = %{{x}}<br>{y} = %{{y}}")
        
        st.plotly_chart(fig, use_container_width=True, height=100)


    def line_chart(df, x, y, text, textposition, color, title, title_x=0.25):

        fig = px.line(df, x=x, y=y, labels={
                      x: '', y: ''}, title=title, text=df[text])

        fig.update_layout(title_x=title_x, title_font_size=22)

        fig.update_traces(line=dict(color=color, width=3.5),
                          marker=dict(symbol='diamond', size=10),
                          texttemplate='%{text}',
                          textfont=dict(size=13.5),
                          textposition=textposition,
                          hovertemplate='%{x}<br>%{y}')

        st.plotly_chart(fig, use_container_width=True, height=100)




   



class Analysis:

    def __init__(self, csv_path):
        # Load CSV file once when creating an instance
        self.df = pd.read_csv(csv_path)

    def type(self):
        df = self.df.groupby('type')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def lease_type(self):
        df = self.df.groupby('lease_type')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def property_size(self):
        df = self.df.groupby('property_size')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def property_age(self):
        df = self.df.groupby('property_age')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def furnishing(self):
        df = self.df.groupby('furnishing')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def facing(self):
        df = self.df.groupby('facing')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def floor(self):
        df = self.df.groupby('floor')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def total_floor(self):
        df = self.df.groupby('total_floor')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def building_type(self):
        df = self.df.groupby('building_type')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def water_supply(self):
        df = self.df.groupby('water_supply')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def negotiable(self):
        df = self.df.groupby('negotiable')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df['negotiable'] = df['negotiable'].apply(lambda x: 'Yes' if x == 1 else 'No')
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def cup_board(self):
        df = self.df.groupby('cup_board')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df['cup_board'] = df['cup_board'].apply(lambda x: f'{int(x)}`')
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def balconies(self):
        df = self.df.groupby('balconies')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df['balconies'] = df['balconies'].apply(lambda x: f'{int(x)}`')
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def parking(self):
        df = self.df.groupby('parking')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def bathroom(self):
        df = self.df.groupby('bathroom')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df['bathroom'] = df['bathroom'].apply(lambda x: f'{int(x)}`')
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def no_of_amenities(self):
        df = self.df.groupby('no_of_amenities')['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df['no_of_amenities'] = df['no_of_amenities'].apply(lambda x: f'{int(x)}`')
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def amenities(self, amenity):
        df = self.df.groupby(amenity)['rent'].mean().reset_index()
        df['rent'] = df['rent'].apply(lambda x: int(x))
        df[amenity] = df[amenity].apply(lambda x: 'Yes' if x == 1 else 'No')
        df = df.sort_values('rent', ascending=False).reset_index(drop=True)
        return df

    def bins(self, df, feature, bins=10):
        # filter 2 columns
        df1 = df[['rent', feature]].copy()
        # calculate bin edges
        bin_edges = pd.cut(df1[feature], bins=bins, labels=False, retbins=True)[1]
        bin_edges[0] = 0
        bin_labels = [f"{int(bin_edges[i])} to {int(bin_edges[i+1])}" for i in range(len(bin_edges)-1)]
        df1['part'] = pd.cut(df1[feature], bins=bin_edges, labels=bin_labels, include_lowest=True)
        df2 = df1.groupby('part')['rent'].mean().reset_index()
        df2 = df2[df2['rent'] > 0]
        df2['rent'] = df2['rent'].apply(lambda x: int(x))
        return df2


class prediction:

    type_dict          = {'RK1':0, 'BHK1':1, 'BHK2':2, 'BHK3':3, 'BHK4':4, 'BHK4PLUS':5}
    lease_type_dict    = {'BACHELOR':1, 'FAMILY':2, 'COMPANY':3, 'ANYONE':4}
    facing_dict        = {'N':1, 'E':2, 'W':3, 'S':4, 'NE':5, 'NW':6, 'SE':7, 'SW':8}
    furnishing_dict    = {'NOT_FURNISHED':0, 'SEMI_FURNISHED':1, 'FULLY_FURNISHED':2}
    parking_dict       = {'NONE':0, 'TWO_WHEELER':1, 'FOUR_WHEELER':2, 'BOTH':3}
    water_supply_dict  = {'CORPORATION':1, 'CORP_BORE':2, 'BOREWELL':3,}
    building_type_dict = {'AP':1, 'IH':2, 'IF':3, 'GC':4}
    binary_dict        = {'Yes':1, 'No':0}
    amenities_dit      = {'gym':'Gym', 'lift':'Lift', 'swimming_pool':'Swimming Pool', 
                        'internet':'Internet', 'ac':'AC', 'club':'Club', 'intercom':'Intercom', 
                        'cpa':'CPA', 'fs':'FS', 'servant':'Servant', 'security':'Security',
                        'sc':'SC', 'gp':'GP', 'park':'Park', 'rwh':'RWH', 'stp':'STP', 
                        'hk':'HK', 'pb':'PB', 'vp':'VP'}

    type_list          = ['RK1', 'BHK1', 'BHK2', 'BHK3', 'BHK4', 'BHK4PLUS']
    facing_list        = ['N', 'E', 'W', 'S', 'NE', 'NW', 'SE', 'SW']
    lease_type_list    = ['BACHELOR', 'FAMILY', 'COMPANY', 'ANYONE']
    furnishing_list    = ['NOT_FURNISHED', 'SEMI_FURNISHED', 'FULLY_FURNISHED']
    parking_list       = ['NONE', 'TWO_WHEELER', 'FOUR_WHEELER', 'BOTH']
    water_supply_list  = ['CORPORATION', 'CORP_BORE', 'BOREWELL']
    building_type_list = ['AP', 'IH', 'IF', 'GC']  # AP-Apartment, IH-Independent House, IF-Inherited House, GC-Guesthouse/Condo
    amenities_list     = ['gym', 'lift', 'swimming_pool', 'internet', 'ac', 
                          'club', 'intercom', 'cpa', 'fs', 'servant', 'security', 
                          'sc', 'gp', 'park', 'rwh', 'stp', 'hk', 'pb', 'vp']
    

def feature_list(feature):
    options = {
        "floor": list(range(0, 51)),        # Floors: 0–50
        "total_floor": list(range(1, 51)),  # Total floors: 1–50
        "cup_board": list(range(0, 11)),    # Cupboards: 0–10
        "balconies": list(range(0, 6)),     # Balconies: 0–5
        "bathroom": list(range(1, 6))       # Bathrooms: 1–5
    }
    return options.get(feature, [])
def predict_rent():
    # Start a form
    with st.form('prediction_form'):
        col1, col2, col3 = st.columns([0.45, 0.1, 0.45])

        with col1:
            activation_date = st.date_input("Activation Date", min_value=date(2017, 1, 1),
                                            max_value=date(2018, 12, 31), value=date(2017, 1, 1))
            latitude = st.number_input("Latitude", min_value=12.90, max_value=12.99, value=12.90)
            longitude = st.number_input("Longitude", min_value=77.50, max_value=80.27, value=77.50)
            type_ = st.selectbox("Property Type", options=prediction.type_list)
            lease_type = st.selectbox("Lease Type", options=prediction.lease_type_list)
            property_size = st.number_input("Property Size", min_value=1, max_value=50000, value=1000)
            property_age = st.number_input("Property Age", min_value=0.0, max_value=400.0, value=0.0)
            furnishing = st.selectbox("Furnishing", options=prediction.furnishing_list)
            facing = st.selectbox("Facing", options=prediction.facing_list)
            floor = st.selectbox("Floor", options=feature_list('floor'))

        with col3:
            total_floor = st.selectbox("Total Floor", options=feature_list('total_floor'))
            building_type = st.selectbox("Building Type", options=prediction.building_type_list)
            water_supply = st.selectbox("Water Supply", options=prediction.water_supply_list)
            negotiable = st.selectbox("Negotiable", options=['Yes', 'No'])
            cup_board = st.selectbox("Cup Board", options=feature_list('cup_board'))
            balconies = st.selectbox("Balconies", options=feature_list('balconies'))
            parking = st.selectbox("Parking", options=prediction.parking_list)
            bathroom = st.selectbox("Bathroom", options=feature_list('bathroom'))
            amenities = st.multiselect("Amenities", options=prediction.amenities_list)

        # Add submit button inside the form
        submit_button = st.form_submit_button("SUBMIT")

    # Run prediction only if submit is clicked
    if submit_button:
        style_submit_button()
        with st.spinner("Processing..."):
           with open(r'regression_model.pkl', 'rb') as f:
            model = pickle.load(f)

            # Encode amenities
            amenity_value = [1 if i in amenities else 0 for i in prediction.amenities_list]
            amenity_value.append(sum(amenity_value))

            data = [
                activation_date.day, activation_date.month, activation_date.year,
                latitude, longitude,
                prediction.type_dict[type_],
                prediction.lease_type_dict[lease_type],
                property_size, property_age,
                prediction.furnishing_dict[furnishing],
                prediction.facing_dict[facing],
                floor, total_floor,
                prediction.building_type_dict[building_type],
                prediction.water_supply_dict[water_supply],
                prediction.binary_dict[negotiable],
                cup_board, balconies,
                prediction.parking_dict[parking],
                bathroom
            ]
            data.extend(amenity_value)

            user_data = np.array([data])
            y_pred = model.predict(user_data)[0]
            rent_price = f"{y_pred:.2f}"

            st.success(f"Predicted Rent: {rent_price}")
            return rent_price


streamlit_config()

with st.sidebar:
    add_vertical_space(2)
    option = option_menu(menu_title='', options=['Data Analysis', 'Prediction', 'Exit'],
                         icons=['database-fill', 'bar-chart-line', 'slash-square', 'sign-turn-right-fill'])
  




if option == 'Data Analysis':
    analysis = Analysis("Dataset/df_final.csv")
    
    # All Data Analysis tabs go here, properly indented
    tab1, tab2, tab3, tab4 = st.tabs(['Type', 'Lease Type', 'Property Size','Property Age'])
    
    with tab1:
        df = analysis.type()
        plotly.vertical_bar_chart(df=df, x='type', y='rent', text='rent', color='#5D9A96', 
                                  title='Property Type wise Average Rent')
   


        
    with tab2:
        df1 = analysis.lease_type()
        plotly.vertical_bar_chart(df=df1, x='lease_type', y='rent', text='rent', 
                                color='#5cb85c', title='Lease Type wise Average Rent')

    with tab3:
        df2 = analysis.property_size()
        df3 = analysis.bins(df=df2, feature='property_size')
        plotly.vertical_bar_chart(df=df3, x='part', y='rent', text='part', color='#5D9A96',
                                title='Property Size wise Average Rent')

    with tab4:
        df4 = analysis.property_age()
        df5 = analysis.bins(df=df4, feature='property_age')
        plotly.vertical_bar_chart(df=df5, x='part', y='rent', text='part', color='#5cb85c',
                                title='Property Age wise Average Rent')


    tab5,tab6,tab7,tab8,tab9,tab10,tab11 = st.tabs(['Furnishing','Building Type',
                                                    'Water Supply', 'Parking','Negotiable',
                                                    'Amenities','Amenities Types'])

    with tab5:
        df6 = analysis.furnishing()
        plotly.pie_chart(df=df6, x='furnishing', y='rent', title_x=0.25, 
                        title='Furnishing wise Average Rent')

    with tab6:
        df12 = analysis.building_type()
        plotly.pie_chart(df=df12, x='building_type', y='rent', title_x=0.28,
                        title='Building Type wise Average Rent')

    with tab7:
        df13 = analysis.water_supply()
        plotly.pie_chart(df=df13, x='water_supply', y='rent', title_x=0.25,
                        title='Water Supply wise Average Rent')

    with tab8:
        df17 = analysis.parking()
        plotly.pie_chart(df=df17, x='parking', y='rent', title_x=0.26,
                        title='Parking wise Average Rent')

    with tab9:
        df14 = analysis.negotiable()
        plotly.pie_chart(df=df14, x='negotiable', y='rent', title_x=0.28,
                        title='Negotiable wise Average Rent')

    with tab10:
        df20 = analysis.no_of_amenities()
        df20['no_of_amenities'] = df20['no_of_amenities'].apply(lambda x: 'Yes' if x!='0`' else 'No')
        df20 = df20.groupby('no_of_amenities').mean().reset_index()
        df20['rent'] = df20['rent'].apply(lambda x: round(x,0))
        plotly.pie_chart(df=df20, x='no_of_amenities', y='rent', title_x=0.28,
                        title='Amenities based Average Rent')

    with tab11:   
        options = ['gym','lift','swimming_pool','internet','ac','club',
                'intercom','cpa','fs','servant','security','sc',
                'gp','park','rwh','stp','hk','pb','vp']
        
        options_dict = {'gym':'Gym', 'lift':'Lift', 'swimming_pool':'Swimming Pool', 
                        'internet':'Internet', 'ac':'AC', 'club':'Club', 
                        'intercom':'Intercom', 'cpa':'CPA', 'fs':'FS', 
                        'servant':'Servant', 'security':'Security', 'sc':'SC', 
                        'gp':'GP', 'park':'Park', 'rwh':'RWH', 'stp':'STP', 
                        'hk':'HK', 'pb':'PB', 'vp':'VP'}
        
        col1,col2,col3 = st.columns([0.33,0.33,0.33])

        with col1:
            amenities = st.selectbox(label='', options=options)
        df21 = analysis.amenities(amenity=amenities)
        plotly.pie_chart(df=df21, x=amenities, y='rent', title_x=0.32,
                 title=f'{options_dict[amenities]} wise Average Rent')



    tab12,tab13,tab14 = st.tabs(['Amenity Count', 'Cup Board','Bathroom'])

    with tab12:
        df19 = analysis.no_of_amenities()
        plotly.line_chart(df=df19, x='no_of_amenities', y='rent', 
                        text='rent', textposition='top right', 
                        color='#5cb85c', title_x=0.30,
                        title='Amenity Count wise Average Rent')

    with tab13:
        df15 = analysis.cup_board()
        plotly.line_chart(df=df15, x='cup_board', y='rent', text='rent', 
                        textposition=['middle right'] + ['top right']*(len(df15['cup_board'])-1), 
                        color='#5D9A96', title_x=0.30,
                        title='Cup Board wise Average Rent')

    with tab14:
        df18 = analysis.bathroom()
        plotly.line_chart(df=df18, x='bathroom', y='rent', text='rent', 
                        textposition='top right', color='#5cb85c', 
                        title_x=0.30, title='Bathroom wise Average Rent')


    tab15,tab16,tab17,tab18 = st.tabs(['Facing', 'Floor','Total Floor','Balconies'])

    with tab15:
        df7 = analysis.facing()
        plotly.vertical_bar_chart(df=df7, x='facing', y='rent', text='facing', 
                                color='#5D9A96', title='Facing wise Average Rent')

    with tab16:
        df8 = analysis.floor()
        df9 = analysis.bins(df=df8, feature='floor', bins=10)
        plotly.vertical_bar_chart(df=df9, x='part', y='rent', text='rent',
                                color='#5cb85c', title='Floor wise Average Rent')

    with tab17:
        df10 = analysis.total_floor()
        df11 = analysis.bins(df=df10, feature='total_floor', bins=10)
        plotly.vertical_bar_chart(df=df11, x='part', y='rent', text='rent',
                                color='#5D9A96', title='Total Floor wise Average Rent')

    with tab18:
        df16 = analysis.balconies()
        plotly.vertical_bar_chart(df=df16, x='balconies', y='rent', text='rent',
                                color='#5cb85c', title='Balconies wise Average Rent')
        


elif option == 'Prediction':
  rent = predict_rent()
  if rent:
        style_prediction()
        st.markdown(f'### <div class="center-text">Predicted Rent = {rent}</div>', 
                    unsafe_allow_html=True)
        st.balloons()

elif option == 'Exit':
    add_vertical_space(2)
    col1, col2, col3 = st.columns([0.20, 0.60, 0.20])
    with col2:
        st.success('#### Thank you for your time. Exiting the application')
        st.balloons()
