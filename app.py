import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# Set page config
st.set_page_config(
    page_title="House Hunters Dashboard",
    layout="wide"
)

# Helper functions for visualizations
def create_price_distribution_chart(df, price_col, title):
    """Create a histogram of price distribution"""
    chart = alt.Chart(df).mark_bar().encode(
        alt.X(price_col, bin=True, title="Price Range"),
        alt.Y('count()', title="Number of Properties"),
        color=alt.Color(price_col, scale=alt.Scale(scheme='blues'))
    ).properties(
        title=title,
        width=600,
        height=400
    )
    return chart

def create_price_comparison_chart(sales_df, rentals_df):
    """Create a comparison chart of sales vs rental prices"""
    sales = sales_df[['PRICE']].copy()
    rentals = rentals_df[['RENT_PER_MONTH']].copy()
    
    # Create bins for comparison
    bins = pd.cut(sales['PRICE'], bins=20)
    sales['PRICE_BIN'] = bins
    rentals['RENT_BIN'] = pd.cut(rentals['RENT_PER_MONTH'], bins=bins)
    
    # Calculate counts
    sales_counts = sales.groupby('PRICE_BIN').size().reset_index(name='SALES_COUNT')
    rentals_counts = rentals.groupby('RENT_BIN').size().reset_index(name='RENTALS_COUNT')
    
    # Merge and create chart
    comparison = pd.merge(sales_counts, rentals_counts, left_on='PRICE_BIN', right_on='RENT_BIN')
    
    chart = alt.Chart(comparison).mark_bar().encode(
        alt.X('PRICE_BIN', title="Price Range"),
        alt.Y('SALES_COUNT', title="Number of Properties"),
        color=alt.Color('variable', scale=alt.Scale(scheme='category10'))
    ).properties(
        title="Sales vs Rentals Price Distribution",
        width=600,
        height=400
    )
    
    return chart

def create_30_year_projection_chart(df):
    """Create a 30-year financial projection chart"""
    # Create sample data for projection
    years = list(range(1, 31))
    
    # Calculate appreciation
    appreciation_rate = 0.03  # 3% annual appreciation
    initial_value = df['PRICE'].mean()
    values = [initial_value * (1 + appreciation_rate) ** y for y in years]
    
    # Create DataFrame
    projection = pd.DataFrame({
        'Year': years,
        'Home Value': values
    })
    
    # Create chart
    chart = alt.Chart(projection).mark_line().encode(
        alt.X('Year', title="Year"),
        alt.Y('Home Value', title="Home Value ($)", scale=alt.Scale(type='log')),
        color=alt.Color(value='blue')
    ).properties(
        title="30-Year Home Value Projection",
        width=600,
        height=400
    )
    
    return chart

def create_rental_projection_chart(df):
    """Create a rental income projection chart"""
    # Create sample data for projection
    years = list(range(1, 31))
    
    # Calculate rental growth
    growth_rate = 0.02  # 2% annual rental growth
    initial_rent = df['RENT_PER_MONTH'].mean()
    rents = [initial_rent * (1 + growth_rate) ** y for y in years]
    
    # Create DataFrame
    projection = pd.DataFrame({
        'Year': years,
        'Monthly Rent': rents
    })
    
    # Create chart
    chart = alt.Chart(projection).mark_line().encode(
        alt.X('Year', title="Year"),
        alt.Y('Monthly Rent', title="Monthly Rent ($)", scale=alt.Scale(type='log')),
        color=alt.Color(value='green')
    ).properties(
        title="30-Year Rental Income Projection",
        width=600,
        height=400
    )
    
    return chart

# Cache data preprocessing
@st.cache_data(show_spinner="Processing data...")
def preprocess_data(df):
    """Pre-process data for charts and maps"""
    if df is None:
        return None
    
    # Convert numeric columns
    numeric_cols = ['PRICE', 'SQUARE_FEET', 'RENT_PER_MONTH', 'RENT_TOTAL_COST_AFTER_30_YEARS',
                   'ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS', 'TOTAL_COSTS_AFTER_30_YEARS',
                   'ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS', 'ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create necessary columns
    if 'ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS' in df.columns:
        df['NET_GAIN_POSITIVE'] = df['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'] > 0
    
    if 'RENT_TOTAL_COST_AFTER_30_YEARS' in df.columns:
        df['COST_CATEGORY'] = pd.cut(
            df['RENT_TOTAL_COST_AFTER_30_YEARS'],
            bins=[0, 1000000, 2000000, float('inf')],
            labels=['<$1M', '$1-2M', '>$2M'],
            include_lowest=True
        )
    
    # Precompute city medians
    if 'PRICE' in df.columns and 'CITY' in df.columns:
        df['CITY_PRICE_MEDIAN'] = df.groupby('CITY')['PRICE'].transform('median')
    
    if 'RENT_PER_MONTH' in df.columns and 'CITY' in df.columns:
        df['CITY_RENT_MEDIAN'] = df.groupby('CITY')['RENT_PER_MONTH'].transform('median')
    
    return df

# Cache charts
@st.cache_data(show_spinner="Creating chart...")
def create_cached_chart(df, chart_type, columns, title=None, width=600, height=400):
    """Create and cache charts with consistent styling"""
    if df is None or not all(col in df.columns for col in columns):
        return None
    
    # Sample data if needed
    if len(df) > 5000:
        df = df.sample(5000, random_state=1)
    
    if chart_type == 'price_distribution':
        chart = alt.Chart(df).mark_bar(color='#7E57C2').encode(
            x=alt.X(columns[0], bin=alt.Bin(maxbins=50), title=columns[0]),
            y=alt.Y('count()', title='Number of Properties')
        ).properties(
            title=title or f"{columns[0]} Distribution",
            width=width,
            height=height
        )
    elif chart_type == 'scatter':
        chart = alt.Chart(df).mark_circle(size=60, opacity=0.4).encode(
            x=alt.X(columns[1], title=columns[1]),
            y=alt.Y(columns[0], title=columns[0], scale=alt.Scale(zero=False)),
            tooltip=columns
        ).properties(
            title=title or f"{columns[0]} vs {columns[1]}",
            width=width,
            height=height
        )
    elif chart_type == 'boxplot':
        chart = alt.Chart(df).mark_boxplot().encode(
            x=alt.X(columns[1], title=columns[1]),
            y=alt.Y(columns[0], title=columns[0], scale=alt.Scale(zero=False))
        ).properties(
            title=title or f"{columns[0]} by {columns[1]}",
            width=width,
            height=height
        )
    elif chart_type == 'pie':
        counts = df[columns[0]].value_counts().reset_index()
        counts.columns = [columns[0], 'COUNT']
        chart = px.pie(counts, values='COUNT', names=columns[0],
                       color=columns[0],
                       color_discrete_map={'Net Gain': '#4caf50', 'Net Loss': '#f44336',
                                          '<$1M': '#4CAF50', '$1-2M': '#FFC107', '>$2M': '#F44336'})
    elif chart_type == 'stacked_area':
        chart = alt.Chart(df).mark_area().encode(
            x=alt.X('YEAR', title="Year"),
            y=alt.Y('VALUE', title="Cumulative Costs ($)"),
            color=alt.Color('CATEGORY', scale=alt.Scale(scheme='category10'))
        ).properties(
            title=title,
            width=width,
            height=height
        )
    elif chart_type == 'line':
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('YEAR', title="Year"),
            y=alt.Y('VALUE', title="Cumulative Costs ($)"),
            color=alt.Color('CATEGORY', scale=alt.Scale(scheme='category10'))
        ).properties(
            title=title,
            width=width,
            height=height
        )
    else:
        return None
    
    return chart

# Load and preprocess data
@st.cache_data(show_spinner="Loading data...")
def load_and_process_data():
    """Load and preprocess data in one step"""
    sales_df, rentals_df = load_data()
    
    if sales_df is not None:
        sales_df = preprocess_data(sales_df)
    if rentals_df is not None:
        rentals_df = preprocess_data(rentals_df)
    
    return sales_df, rentals_df

# Load data with caching and progress
@st.cache_data(show_spinner="Loading data...")
def load_data():
    try:
        # Load the CSV data files
        sales = pd.read_csv("data/home_sales.csv")
        rentals = pd.read_csv("data/home_rentals.csv")
        
        # Clean price data
        def clean_price(x):
            if pd.isna(x):
                return np.nan
            try:
                # Remove special characters and convert to float
                price_str = str(x).replace('$', '').replace(',', '').replace(' ', '').strip()
                if not price_str:  # Handle empty strings
                    return np.nan
                return float(price_str)
            except:
                return np.nan

        sales['PRICE'] = sales['PRICE'].apply(clean_price)
        rentals['RENT_PER_MONTH'] = rentals['RENT_PER_MONTH'].apply(clean_price)
        
        # Sample data if too large
        if len(sales) > 10000:
            sales = sales.sample(10000, random_state=42)
        if len(rentals) > 10000:
            rentals = rentals.sample(10000, random_state=42)
        
        return sales, rentals
    except FileNotFoundError:
        st.error("Error: Data files not found. Please ensure 'data/home_sales.csv' and 'data/home_rentals.csv' exist.")
        return None, None
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Create 30-year projection data
@st.cache_data(show_spinner="Creating 30-year projections...")
def create_30_year_projections(sales_df, rentals_df):
    """Create 30-year cost projections for buying and renting"""
    if sales_df is None or rentals_df is None:
        return None, None
    
    # Get median values
    median_price = sales_df['PRICE'].median()
    median_rent = rentals_df['RENT_PER_MONTH'].median()
    
    # Create year range
    years = list(range(1, 31))
    
    # Buying costs
    buying_costs = []
    for year in years:
        # Mortgage (6% interest, 20% down)
        down_payment = median_price * 0.2
        loan_amount = median_price * 0.8
        monthly_payment = (loan_amount * 0.06/12) / (1 - (1 + 0.06/12)**(-360))
        total_mortgage = monthly_payment * 12 * year
        
        # Property taxes (1.2%)
        property_taxes = median_price * 0.012 * year
        
        # Maintenance (1%)
        maintenance = median_price * 0.01 * year
        
        # Insurance (0.5%)
        insurance = median_price * 0.005 * year
        
        # Home value appreciation (3%)
        appreciation = median_price * ((1 + 0.03)**year - 1)
        
        buying_costs.append({
            'YEAR': year,
            'CATEGORY': 'Mortgage',
            'VALUE': total_mortgage
        })
        buying_costs.append({
            'YEAR': year,
            'CATEGORY': 'Property Taxes',
            'VALUE': property_taxes
        })
        buying_costs.append({
            'YEAR': year,
            'CATEGORY': 'Maintenance',
            'VALUE': maintenance
        })
        buying_costs.append({
            'YEAR': year,
            'CATEGORY': 'Insurance',
            'VALUE': insurance
        })
        buying_costs.append({
            'YEAR': year,
            'CATEGORY': 'Appreciation',
            'VALUE': -appreciation  # Negative value for stacking
        })
    
    # Renting costs
    renting_costs = []
    for year in years:
        # Rent with 2.5% annual increase
        total_rent = median_rent * 12 * (1 + 0.025)**(year-1)
        
        renting_costs.append({
            'YEAR': year,
            'CATEGORY': 'Rent',
            'VALUE': total_rent
        })
        
        # Rent inflation
        if year > 1:
            inflation = total_rent - (median_rent * 12 * (1 + 0.025)**(year-2))
            renting_costs.append({
                'YEAR': year,
                'CATEGORY': 'Rent Inflation',
                'VALUE': inflation
            })
    
    return pd.DataFrame(buying_costs), pd.DataFrame(renting_costs)

# Create city affordability ranking
@st.cache_data(show_spinner="Calculating city affordability...")
def create_city_affordability(sales_df, rentals_df):
    """Create city affordability ranking based on median salary vs housing costs"""
    if sales_df is None or rentals_df is None:
        return None
    
    # Get city data
    city_data = sales_df[['CITY', 'CITY_PRICE_MEDIAN']].merge(
        rentals_df[['CITY', 'CITY_RENT_MEDIAN']],
        on='CITY',
        how='inner'
    )
    
    # Assume median salary is 3x median rent
    city_data['MEDIAN_SALARY'] = city_data['CITY_RENT_MEDIAN'] * 3
    
    # Calculate housing cost ratios
    city_data['BUYING_RATIO'] = city_data['CITY_PRICE_MEDIAN'] / city_data['MEDIAN_SALARY']
    city_data['RENTING_RATIO'] = city_data['CITY_RENT_MEDIAN'] / city_data['MEDIAN_SALARY']
    
    # Calculate overall affordability score
    city_data['AFFORDABILITY_SCORE'] = (city_data['BUYING_RATIO'] + city_data['RENTING_RATIO']) / 2
    
    # Rank cities by affordability
    city_data = city_data.sort_values('AFFORDABILITY_SCORE')
    
    return city_data

# Create city performance metrics
@st.cache_data(show_spinner="Calculating city performance metrics...")
def create_city_performance(sales_df):
    """Calculate city performance metrics for home value growth and ROI"""
    if sales_df is None:
        return None, None
    
    # Ensure all required columns exist and are numeric
    required_cols = ['PRICE', 'RENT_PER_MONTH', 'SQUARE_FEET', 'CITY']
    for col in required_cols:
        if col not in sales_df.columns:
            sales_df[col] = 0  # Add missing columns with default value
        sales_df[col] = pd.to_numeric(sales_df[col], errors='coerce')
    
    # Calculate derived metrics
    sales_df['DOWNPAYMENT_20PERCENT'] = sales_df['PRICE'] * 0.2
    sales_df['MONTHLY_MORTGAGE'] = (sales_df['PRICE'] * 0.8 * 0.04) / 12  # 4% interest rate
    sales_df['MONTHLY_COSTS'] = sales_df['MONTHLY_MORTGAGE'] + (sales_df['PRICE'] * 0.01) / 12  # 1% property tax
    sales_df['TOTAL_COSTS_AFTER_30_YEARS'] = sales_df['MONTHLY_COSTS'] * 360  # 30 years
    
    # Calculate home value appreciation
    sales_df['ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS'] = sales_df['PRICE'] * (1 + 0.03) ** 30  # 3% annual appreciation
    
    # Calculate net value gain
    sales_df['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'] = (sales_df['ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS'] - 
                                                         sales_df['PRICE']) - sales_df['TOTAL_COSTS_AFTER_30_YEARS']
    
    # Filter for valid data
    valid_df = sales_df.dropna(subset=['ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS', 
                                      'ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS',
                                      'TOTAL_COSTS_AFTER_30_YEARS',
                                      'DOWNPAYMENT_20PERCENT'])
    
    # Get top 10 cities by number of homes
    top_cities = valid_df['CITY'].value_counts().head(10).index
    
    # Calculate metrics for top cities
    city_metrics = valid_df[valid_df['CITY'].isin(top_cities)].groupby('CITY').agg({
        'ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS': 'mean',
        'TOTAL_COSTS_AFTER_30_YEARS': 'mean',
        'ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS': 'mean'
    }).reset_index()
    
    # Calculate ROI metrics
    roi_metrics = valid_df[valid_df['CITY'].isin(top_cities)].groupby('CITY').agg({
        'ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS': 'mean',
        'DOWNPAYMENT_20PERCENT': 'mean'
    }).reset_index()
    
    roi_metrics['ROI_PERCENTAGE'] = (roi_metrics['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'] / 
                                    roi_metrics['DOWNPAYMENT_20PERCENT']) * 100
    
    return city_metrics, roi_metrics

# Create rental metrics
@st.cache_data(show_spinner="Calculating rental metrics...")
def create_rental_metrics(rentals_df):
    """Calculate rental metrics for top and bottom cities"""
    if rentals_df is None:
        return None, None
    
    # Calculate average monthly rent by city
    avg_rent = rentals_df.groupby('CITY')['RENT_PER_MONTH'].mean().reset_index()
    
    # Get top 10 cities by average rent
    top_rent = avg_rent.nlargest(10, 'RENT_PER_MONTH')
    bottom_rent = avg_rent.nsmallest(10, 'RENT_PER_MONTH')
    
    return top_rent, bottom_rent

# Create REIT comparison metrics
@st.cache_data(show_spinner="Calculating REIT comparison metrics...")
def create_reit_comparison(sales_df):
    """Calculate REIT vs Homeownership returns"""
    if sales_df is None:
        return None
    
    valid_df = sales_df.dropna(subset=['ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS',
                                      'ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS',
                                      'TOTAL_INTEREST_30_YEARS',
                                      'DOWNPAYMENT_20PERCENT'])
    
    return valid_df.groupby('CITY').agg({
        'ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS': 
            lambda x: ((x / valid_df['DOWNPAYMENT_20PERCENT']).mean() * 100),
        'ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS': 
            lambda x: (((x - valid_df['TOTAL_INTEREST_30_YEARS']) / valid_df['DOWNPAYMENT_20PERCENT']).mean() * 100)
    }).reset_index()

# Create ROI comparison chart
@st.cache_data(show_spinner="Creating ROI comparison chart...")
def create_roi_comparison_chart(sales_df):
    """Create a comparison chart of ROI between top and bottom cities"""
    if sales_df is None:
        return None
    
    # Calculate ROI metrics
    valid_df = sales_df.dropna(subset=['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS', 'DOWNPAYMENT_20PERCENT'])
    roi_metrics = valid_df.groupby('CITY').apply(
        lambda x: (x['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'] / x['DOWNPAYMENT_20PERCENT']).mean() * 100,
        include_groups=False
    ).reset_index(name='ROI_PERCENTAGE')
    
    # Split into top and bottom 10
    top_cities = roi_metrics.nlargest(10, 'ROI_PERCENTAGE')
    bottom_cities = roi_metrics.nsmallest(10, 'ROI_PERCENTAGE')
    
    # Create combined DataFrame
    combined = pd.concat([
        top_cities.assign(CATEGORY='Top 10'),
        bottom_cities.assign(CATEGORY='Bottom 10')
    ])
    
    # Create bar chart
    chart = alt.Chart(combined).mark_bar().encode(
        x=alt.X('CITY', sort='-y', title='City'),
        y=alt.Y('ROI_PERCENTAGE:Q', title='ROI Percentage'),
        color=alt.Color('CATEGORY:N', title='Category')
    ).properties(
        title=alt.TitleParams(
            'Return on Investment After 30 Years: Top vs Bottom Cities',
            subtitle='ROI calculated as net gain divided by downpayment (20%)'
        ),
        width=800,
        height=400
    )
    
    return chart

# Sidebar layout
st.sidebar.title("House Hunters Dashboard")

# Navigation
page = st.sidebar.selectbox(
    "Select Page",
    ["Dashboard", "Investment Insights", "Predictions", "About"]
)

# Main content area
if page == "Dashboard":
    st.title("Real Estate Market Dashboard")
    # ... rest of the dashboard content ...

# Add developers section at the bottom
with st.sidebar.expander("👥 Developers", expanded=True):
    
    st.markdown("""
    <style>
    .developer-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 15px;
        width: 100%;
    }
    .developer-name {
        font-size: 16px;
        font-weight: bold;
        color: #1a1a1a;
    }
    .developer-role {
        font-size: 14px;
        color: #666;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display developers vertically
    st.markdown("""
    <div class="developer-card">
        <div class="developer-name">Aung Nyein Chan Kyaw</div>
        <div class="developer-role">Team Lead | Data Concierge</div>
    </div>
    
    <div class="developer-card">
        <div class="developer-name">Ashwinth Reddy Kondapalli</div>
        <div class="developer-role">Data Analyst | Tech Guy</div>
    </div>
    """, unsafe_allow_html=True)
    
        

# Main content area
if page == "Dashboard":
    
    
    # Load and process data once
    with st.spinner('Loading data...'):
        sales_df, rentals_df = load_and_process_data()

        if sales_df is not None:
            # Create tabs
            tab_sales, tab_rentals = st.tabs(["Sales", "Rentals"])

            with tab_sales:
                # Sub-tabs for Sales
                sales_map_tab, sales_chart_tab, sales_proj_tab = st.tabs(["Location Maps", "Charts", "30-Year Projections"])
                
                with sales_map_tab:
                    # Load and display sales cluster map
                    st.subheader("Sales Location Map")
                    try:
                        with open('Outputs/sales_cluster_map.html', 'r') as f:
                            sales_map_html = f.read()
                        st.components.v1.html(sales_map_html, height=600)
                        st.caption("Cluster map showing the distribution of home sales across different locations.")
                    except Exception as e:
                        st.warning(f"Error loading sales cluster map: {str(e)}")

                with sales_chart_tab:
                    # Price Distribution Chart
                    if 'PRICE' in sales_df.columns:
                        try:
                            with st.spinner("Loading price distribution chart..."):
                                price_chart = create_cached_chart(sales_df, 'price_distribution', ['PRICE'],
                                                                title="Home Sale Price Distribution")
                                if price_chart:
                                    st.altair_chart(price_chart, use_container_width=True)
                                    st.caption("Distribution of home sale prices across the market. Higher prices are typically found in coastal and urban areas.")
                        except Exception as e:
                            st.warning(f"Error creating price distribution chart: {str(e)}")
                    else:
                        st.warning("PRICE column not found in sales data")

                    # Scatter: Price vs Square Feet
                    if 'PRICE' in sales_df.columns and 'SQUARE_FEET' in sales_df.columns:
                        try:
                            with st.spinner("Loading price vs square feet chart..."):
                                scatter_chart = create_cached_chart(sales_df, 'scatter', ['PRICE', 'SQUARE_FEET', 'BEDS', 'BATHS', 'CITY'],
                                                                   title="Home Price vs Living Area")
                                if scatter_chart:
                                    st.altair_chart(scatter_chart, use_container_width=True)
                                    st.caption("Relationship between home size and sale price. Larger homes generally command higher prices, though there's significant variation.")
                        except Exception as e:
                            st.warning(f"Error creating price vs square feet chart: {str(e)}")
                    else:
                        st.warning("Required columns not found for price vs square feet chart")

                with sales_proj_tab:
                    # Stacked area chart for buying costs
                    st.subheader("30-Year Buying Cost Breakdown")
                    
                    if sales_df is not None:
                        try:
                            with st.spinner("Creating cost breakdown chart..."):
                                buying_costs, _ = create_30_year_projections(sales_df, rentals_df)
                                if buying_costs is not None:
                                    chart = create_cached_chart(
                                        buying_costs,
                                        'stacked_area',
                                        ['YEAR', 'VALUE', 'CATEGORY'],
                                        title="Cumulative Costs Over 30 Years (Buying)",
                                        width=800,
                                        height=400
                                    )
                                    if chart:
                                        st.altair_chart(chart, use_container_width=True)
                                        st.caption("""
                                        Breakdown of cumulative costs when buying a home:
                                        - Mortgage payments (includes principal and interest)
                                        - Property taxes (1.2%)
                                        - Maintenance (1%)
                                        - Insurance (0.5%)
                                        - Home value appreciation (3%)
                                        """)
                        except Exception as e:
                            st.warning(f"Error creating buying cost breakdown chart: {str(e)}")

            with tab_rentals:
                # Sub-tabs for Rentals
                rent_map_tab, rent_chart_tab, rent_proj_tab = st.tabs(["Location Maps", "Charts", "30-Year Projections"])
                
                with rent_map_tab:
                    # Load and display rentals cluster map
                    st.subheader("Rentals Location Map")
                    try:
                        with open('Outputs/rentals_cluster_map.html', 'r') as f:
                            rentals_map_html = f.read()
                        st.components.v1.html(rentals_map_html, height=600)
                        st.caption("Cluster map showing the distribution of rental properties across different locations.")
                    except Exception as e:
                        st.warning(f"Error loading rentals cluster map: {str(e)}")

                with rent_chart_tab:
                    # Rent Distribution Chart
                    if 'RENT_PER_MONTH' in rentals_df.columns:
                        try:
                            with st.spinner("Loading rent distribution chart..."):
                                rent_chart = create_cached_chart(rentals_df, 'price_distribution', ['RENT_PER_MONTH'],
                                                               title="Monthly Rent Distribution")
                                if rent_chart:
                                    st.altair_chart(rent_chart, use_container_width=True)
                                    st.caption("Distribution of monthly rental prices across the market. Higher rents are typically found in urban areas.")
                        except Exception as e:
                            st.warning(f"Error creating rent distribution chart: {str(e)}")
                    else:
                        st.warning("RENT_PER_MONTH column not found in rentals data")

                    # Scatter: Rent vs Square Feet
                    if 'RENT_PER_MONTH' in rentals_df.columns and 'SQUARE_FEET' in rentals_df.columns:
                        try:
                            with st.spinner("Loading rent vs square feet chart..."):
                                rent_scatter = create_cached_chart(rentals_df, 'scatter', ['RENT_PER_MONTH', 'SQUARE_FEET', 'BEDS', 'BATHS', 'CITY'],
                                                                title="Monthly Rent vs Living Area")
                                if rent_scatter:
                                    st.altair_chart(rent_scatter, use_container_width=True)
                                    st.caption("Relationship between rental size and monthly rent. Larger rentals generally command higher rents, though there's significant variation.")
                        except Exception as e:
                            st.warning(f"Error creating rent vs square feet chart: {str(e)}")
                    else:
                        st.warning("Required columns not found for rent vs square feet chart")

                with rent_proj_tab:
                    # Stacked area chart for renting costs
                    st.subheader("30-Year Renting Cost Breakdown")
                    
                    if rentals_df is not None:
                        try:
                            with st.spinner("Creating cost breakdown chart..."):
                                _, renting_costs = create_30_year_projections(sales_df, rentals_df)
                                if renting_costs is not None:
                                    chart = create_cached_chart(
                                        renting_costs,
                                        'stacked_area',
                                        ['YEAR', 'VALUE', 'CATEGORY'],
                                        title="Cumulative Costs Over 30 Years (Renting)",
                                        width=800,
                                        height=400
                                    )
                                    if chart:
                                        st.altair_chart(chart, use_container_width=True)
                                        st.caption("""
                                        Breakdown of cumulative costs when renting:
                                        - Monthly rent payments
                                        - Rent inflation (2.5% annual increase)
                                        """)
                        except Exception as e:
                            st.warning(f"Error creating renting cost breakdown chart: {str(e)}")

            
elif page == "Investment Insights":
    st.title("Investment Insights: Buy vs Rent in California (2025)")
    st.markdown("**Analyzing 30-year projections, rental affordability, and property value trends.**")
    
    # Load and process data
    with st.spinner('Loading data...'):
        sales_df, rentals_df = load_and_process_data()
    
    if sales_df is None or rentals_df is None:
        st.error("Error loading or processing data. Please check the data files.")
        st.stop()
    
    # Create sub-tabs for different sections
    investment_tabs = st.tabs(["Key Metrics", "Investment ROI", "Rental Affordability", "Properties"])
    
    with investment_tabs[0]:
        # Key Metrics Section
        st.subheader("Key Market Metrics")
        
        try:
            # Use fixed numbers for total listings
            total_sales = 57530
            total_rentals = 15980
            
            # Calculate averages from data
            avg_price = sales_df['PRICE'].mean()
            avg_rent = rentals_df['RENT_PER_MONTH'].mean()
            
            # Create metric columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Sales Listings", f"{total_sales:,}")
            with col2:
                st.metric("Total Rental Listings", f"{total_rentals:,}")
            with col3:
                st.metric("Average Home Sale Price", f"${int(avg_price):,}")
            with col4:
                st.metric("Average Monthly Rent", f"${int(avg_rent):,}")
        except Exception as e:
            st.warning(f"Error calculating metrics: {str(e)}")
            
            # Top Cities by Listings
        st.subheader("Top 10 Cities by Number of Listings")
        try:
            if 'CITY' in sales_df.columns:
                city_counts = sales_df['CITY'].value_counts().head(10).reset_index()
                city_counts.columns = ['CITY', 'COUNT']
                
                chart = alt.Chart(city_counts).mark_bar().encode(
                    x=alt.X('CITY:N', sort='-y', title='City'),
                    y=alt.Y('COUNT:Q', title='Number of Listings'),
                    color=alt.Color('CITY:N', scale=alt.Scale(scheme='category20'))
                ).properties(
                    title="Top 10 Cities by Number of Listings",
                    width=800,
                    height=400
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("City data not available in the dataset")
        except Exception as e:
            st.warning(f"Error creating city listings chart: {str(e)}")

    with investment_tabs[1]:
        # Investment ROI Section
        st.subheader("Investment ROI Analysis")
        
        # Display images vertically
        st.image("images/Return on Investment After 30 Years (Top 10 Cities).png",
                caption="Top 10 Cities by ROI",
                use_column_width=True)
        
        st.image("images/Return on Investment After 30 Years (Worst 10 Cities).png",
                caption="Worst 10 Cities by ROI",
                use_column_width=True)
        
        # Add explanation
        st.caption("""
        ROI Calculation:
        (Net Gain After 30 Years / Initial 20% Downpayment) × 100%
        
        * Positive values indicate profitable investments
        * Negative values indicate potential losses
        * Higher ROI indicates better investment potential
        """)
        
    
    
    with investment_tabs[2]:
        
        # Rent Distribution
        st.subheader("Rent Distribution by City")
        try:
            if 'CITY' in rentals_df.columns:
                # Calculate rent metrics
                rent_metrics = rentals_df.groupby('CITY').agg({
                    'RENT_PER_MONTH': ['mean', 'median', 'count']
                }).reset_index()
                rent_metrics.columns = ['CITY', 'MEAN_RENT', 'MEDIAN_RENT', 'COUNT']
                
                # Split into top and bottom cities
                top_rent = rent_metrics.nlargest(10, 'MEAN_RENT')
                bottom_rent = rent_metrics.nsmallest(10, 'MEAN_RENT')
                
                # Create charts side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    chart1 = alt.Chart(top_rent).mark_bar().encode(
                        x=alt.X('CITY:N', sort='-y', title='City', axis=alt.Axis(labelAngle=30)),
                        y=alt.Y('MEAN_RENT:Q', title='Average Monthly Rent ($)'),
                        tooltip=['CITY', 'MEAN_RENT', 'COUNT']
                    ).properties(
                        title="Top 10 Cities with Highest Rents",
                        width=400,
                        height=400
                    )
                    st.altair_chart(chart1, use_container_width=True)
                
                with col2:
                    chart2 = alt.Chart(bottom_rent).mark_bar().encode(
                        x=alt.X('CITY:N', sort='-y', title='City', axis=alt.Axis(labelAngle=30)),
                        y=alt.Y('MEAN_RENT:Q', title='Average Monthly Rent ($)'),
                        tooltip=['CITY', 'MEAN_RENT', 'COUNT']
                    ).properties(
                        title="Top 10 Cities with Lowest Rents",
                        width=400,
                        height=400
                    )
                    st.altair_chart(chart2, use_container_width=True)
                
                # Add explanation
                st.caption("""
                * Red bars represent cities with highest rents
                * Green bars represent cities with lowest rents
                * Numbers show average monthly rent and number of listings
                """)
            else:
                st.warning("City data not available for rent distribution")
        except Exception as e:
            st.warning(f"Error creating rent distribution charts: {str(e)}")
    
    with investment_tabs[3]:
        # Top Properties Table
        st.subheader("Top 20 Most Expensive Properties")
        try:
            if 'PRICE' in sales_df.columns and 'BEDS' in sales_df.columns and 'BATHS' in sales_df.columns:
                top_properties = sales_df.nlargest(20, 'PRICE')[[
                    'FULL_ADDRESS', 'PRICE', 'BEDS', 'BATHS', 'CITY'
                ]]
                st.dataframe(
                    top_properties,
                    column_config={
                        'PRICE': st.column_config.NumberColumn(
                            "Price",
                            format="$%d",
                            width='medium'
                        ),
                        'BEDS': st.column_config.NumberColumn(
                            "Bedrooms",
                            width='small'
                        ),
                        'BATHS': st.column_config.NumberColumn(
                            "Bathrooms",
                            width='small'
                        )
                    },
                    hide_index=True
                )
            else:
                st.warning("Required columns for property insights not available")
        except Exception as e:
            st.warning(f"Error creating property insights table: {str(e)}")

elif page == "Predictions":
    st.title("Price Predictions")
    
    # Load KNN model data
    def load_knn_data():
        try:
            sales_df = pd.read_csv('data/home_sales.csv')
            rentals_df = pd.read_csv('data/home_rentals.csv')
            
            # Clean both datasets
            for df in [sales_df, rentals_df]:
                df.columns = df.columns.str.strip()
                
                # Clean currency columns
                def clean_currency(x):
                    if pd.notna(x) and isinstance(x, (str, float)):
                        try:
                            price_str = str(x).replace('$', '').replace(',', '').replace(' ', '').strip()
                            if not price_str:  
                                return np.nan
                            return float(price_str)
                        except:
                            return np.nan

                if 'PRICE' in df.columns:
                    df['PRICE'] = df['PRICE'].apply(clean_currency)
                if 'RENT_PER_MONTH' in df.columns:
                    df['RENT_PER_MONTH'] = df['RENT_PER_MONTH'].apply(clean_currency)
                if 'SQUARE_FEET' in df.columns:
                    df['SQUARE_FEET'] = df['SQUARE_FEET'].apply(clean_currency)
            
            # Sample data if too large
            if len(sales_df) > 10000:
                sales_df = sales_df.sample(10000, random_state=42)
            if len(rentals_df) > 10000:
                rentals_df = rentals_df.sample(10000, random_state=42)
            
            return sales_df, rentals_df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None, None

    sales_df, rentals_df = load_knn_data()
    
    if sales_df is not None and rentals_df is not None:
        # Get unique cities from both datasets
        cities = pd.concat([sales_df['CITY'], rentals_df['CITY']]).unique()
        
        # User inputs
        col1, col2 = st.columns(2)
        with col1:
            selected_city = st.selectbox("Select a city", cities)
        with col2:
            property_type = st.selectbox("Select property type", ["Sales", "Rentals"])
            
        if st.button("Get Prediction"):
            with st.spinner('Calculating prediction...'):
                # Get relevant data based on property type
                if property_type == "Sales":
                    df = sales_df
                    target = 'PRICE'
                    title = "Predicted Sale Price"
                else:
                    df = rentals_df
                    target = 'RENT_PER_MONTH'
                    title = "Predicted Monthly Rent"
                
                # Filter data for selected city
                city_data = df[df['CITY'] == selected_city]
                
                # Prepare features and target
                features = ['BEDS', 'BATHS', 'SQUARE_FEET']
                
                X = city_data[features]
                y = city_data[target]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train KNN model
                knn_model = KNeighborsRegressor(n_neighbors=5)
                knn_model.fit(X_train_scaled, y_train)
                
                # Make prediction
                y_pred = knn_model.predict(X_test_scaled)
                
                # Calculate cumulative statistics
                avg_pred = np.mean(y_pred)
                min_pred = np.min(y_pred)
                max_pred = np.max(y_pred)
                
                # Display results in a clean format
                st.subheader(f"{title} for {selected_city}")
                
                # Create metrics in a single row
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Price", f"${int(avg_pred):,}")
                with col2:
                    st.metric("Minimum Price", f"${int(min_pred):,}")
                with col3:
                    st.metric("Maximum Price", f"${int(max_pred):,}")
                
                # Show price distribution chart
                st.subheader("Price Distribution")
                price_dist = pd.DataFrame({
                    'Price': y_pred,
                    'Type': property_type
                })
                
                chart = alt.Chart(price_dist).mark_bar().encode(
                    alt.X('Price', bin=True, title=f"Predicted {property_type} Price"),
                    alt.Y('count()', title="Number of Properties"),
                    color=alt.Color('Type', scale=alt.Scale(scheme='category10'))
                ).properties(
                    width=600,
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)
                
                # Add interpretation
                st.subheader("Interpretation")
                st.write(f"The model predicts that in {selected_city}, the typical {property_type.lower()} price is around ${int(avg_pred):,}.")
                st.write(f"Prices can range from ${int(min_pred):,} to ${int(max_pred):,}, depending on property characteristics.")
                st.write("These predictions are based on recent market data and similar properties.")

elif page == "About":
    st.title("About")
    st.write("""    Welcome to the **House Hunters Dashboard** — a data-powered real estate decision support tool developed for first-time homebuyers and investors.

    This application helps users analyze the long-term financial implications of buying vs renting in California, using actual property data, predictive models, and market simulations.

    ### Key Features:
    - **Interactive Maps** of home sales and rentals across California
    - **30-Year Financial Projections** based on interest rates, taxes, insurance, and appreciation
    - **Rental Price Predictions** using K-Nearest Neighbors (KNN) machine learning
    - **Buy vs Rent Comparison** with projected REIT investment returns
    - **Custom Filters** by city, bedrooms, bathrooms, and more

    ### Technology Stack:
    - **Python, Streamlit, scikit-learn** for predictive modeling
    - **BigQuery and SQL** for scalable data handling
    - **Grafana** for high-performance dashboards
    - **Folium, Plotly, and Matplotlib** for geospatial and financial visualizations

    ### Project Goal:
    Empower users to make smarter real estate decisions with transparent data, predictive models, and visual insights.

    Built by:
    - Aung Nyein Chan Kyaw (Team Lead | Data Concierge)
    - Ashwinth Reddy Kondapalli (Data Analyst | Tech Guy)
                
    """)

elif page == "Developers":
    st.title("Development Team")
    
    # Create two columns for team members
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Person 1")
        st.subheader("Lead Developer / Data Strategist")
        st.write("""
        - Lead the development and architecture of the dashboard
        - Designed the data processing pipeline
        - Implemented the interactive visualization features
        - Optimized performance and user experience
        """)
    
    with col2:
        st.header("Person 2")
        st.subheader("Frontend Developer / UI/UX Specialist")
        st.write("""
        - Created the modern and intuitive user interface
        - Implemented responsive design
        - Added interactive features and animations
        - Ensured cross-browser compatibility
        """)
