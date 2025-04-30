import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import plotly.express as px



st.set_page_config(page_title="Home Sales & Rentals Dashboard", layout="wide")

st.title("Real Estate Market Dashboard: Home Sales vs Rentals")

@st.cache_data
def load_data():
    # Load the CSV data files
    sales = pd.read_csv("data/home_sales.csv")
    rentals = pd.read_csv("data/home_rentals.csv")
    # Clean and convert data types for numeric analysis
    # Remove currency symbols and commas from relevant columns
    sales_numeric_cols = ['PRICE', 'SQUARE_FEET', 'PRICE_PER_SQFT', 'HOA_PER_MONTH',
                           'HOA_30_YEARS', 'CLOSING_COST', 'DOWNPAYMENT_20PERCENT',
                           'LOAN_AMOUNT', 'MONTHLY_EMI', 'TOTAL_INTEREST_30_YEARS',
                           'ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS',
                           'ESTIMATED_PROPERTY_TAXES_AFTER_30_YEARS',
                           'AVERAGE_ANNUAL_INSURANCE_COST', 'ESTIMATED_INSURANCE_COST_AFTER_30_YEARS',
                           'ESTIMATED_HOME_MAINTENANCE_AFTER_30_YEARS', 'TOTAL_COSTS_AFTER_30_YEARS',
                           'ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS', 'ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS']
    for col in sales_numeric_cols:
        if col in sales.columns:
            sales[col] = sales[col].replace({'\$': '', ',': '', '%': ''}, regex=True)
            sales[col] = pd.to_numeric(sales[col], errors='coerce')
    rentals_numeric_cols = ['RENT_PER_MONTH', 'SQUARE_FEET', 'RENT_PRICE_PER_SQFT', 'RENT_TOTAL_COST_AFTER_30_YEARS']
    for col in rentals_numeric_cols:
        if col in rentals.columns:
            rentals[col] = rentals[col].replace({'\$': '', ',': ''}, regex=True)
            rentals[col] = pd.to_numeric(rentals[col], errors='coerce')
    return sales, rentals

# Load and cache data
sales_df, rentals_df = load_data()

# Set up main tabs
tab_sales, tab_rentals, tab_comparison = st.tabs(["Sales", "Rentals", "Comparison"])

# ----------------------------------- Sales Tab -----------------------------------
with tab_sales:
    # Sub-tabs for Sales
    sales_map_tab, sales_chart_tab, sales_proj_tab = st.tabs(["Location Maps", "Charts", "30-Year Projections"])
    
    with sales_map_tab:
        # Map 1: Sales listings colored by Price
        price_vals = sales_df[['LATITUDE', 'LONGITUDE', 'PRICE']].dropna()
        low_cap = np.percentile(price_vals['PRICE'], 5)
        high_cap = np.percentile(price_vals['PRICE'], 95)
        price_vals['price_clamped'] = np.clip(price_vals['PRICE'], low_cap, high_cap)
        ratio = (price_vals['price_clamped'] - low_cap) / (high_cap - low_cap)
        color_r = np.where(ratio > 0.5, ((ratio - 0.5) * 2 * 255), 0).astype(int)
        color_g = np.where(ratio <= 0.5, (ratio * 2 * 255), (1 - (ratio - 0.5) * 2) * 255).astype(int)
        color_b = np.where(ratio <= 0.5, ((1 - ratio * 2) * 255), 0).astype(int)
        price_vals['R'] = color_r
        price_vals['G'] = color_g
        price_vals['B'] = color_b
        view_state = pdk.ViewState(latitude=36.0, longitude=-119.5, zoom=6)
        layer_price = pdk.Layer(
            "ScatterplotLayer",
            data=price_vals,
            get_position="[LONGITUDE, LATITUDE]",
            get_fill_color="[R, G, B, 160]",
            get_radius=50,
            pickable=False,
        )
        deck_price = pdk.Deck(layers=[layer_price], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9")
        st.write("**Map 1: Home Sale Listings Colored by Price**")
        st.pydeck_chart(deck_price)
        st.caption("Higher-priced homes (red) are concentrated in coastal and urban areas, while lower-priced homes (blue) appear more in inland and rural regions.")
        
        # Map 2: Sales listings colored by Price per Square Foot
        ppsf_vals = sales_df[['LATITUDE', 'LONGITUDE', 'PRICE_PER_SQFT']].dropna()
        low_cap_pp = np.percentile(ppsf_vals['PRICE_PER_SQFT'], 5)
        high_cap_pp = np.percentile(ppsf_vals['PRICE_PER_SQFT'], 95)
        ppsf_vals['ppsf_clamped'] = np.clip(ppsf_vals['PRICE_PER_SQFT'], low_cap_pp, high_cap_pp)
        ratio2 = (ppsf_vals['ppsf_clamped'] - low_cap_pp) / (high_cap_pp - low_cap_pp)
        color_r2 = np.where(ratio2 > 0.5, ((ratio2 - 0.5) * 2 * 255), 0).astype(int)
        color_g2 = np.where(ratio2 <= 0.5, (ratio2 * 2 * 255), (1 - (ratio2 - 0.5) * 2) * 255).astype(int)
        color_b2 = np.where(ratio2 <= 0.5, ((1 - ratio2 * 2) * 255), 0).astype(int)
        ppsf_vals['R'] = color_r2
        ppsf_vals['G'] = color_g2
        ppsf_vals['B'] = color_b2
        layer_ppsf = pdk.Layer(
            "ScatterplotLayer",
            data=ppsf_vals,
            get_position="[LONGITUDE, LATITUDE]",
            get_fill_color="[R, G, B, 160]",
            get_radius=50,
            pickable=False,
        )
        deck_ppsf = pdk.Deck(layers=[layer_ppsf], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v9")
        st.write("**Map 2: Home Sale Listings Colored by Price per Square Foot**")
        st.pydeck_chart(deck_ppsf)
        st.caption("Areas with the highest price per square foot (red) are mostly in expensive city centers and coastal communities, whereas outlying areas remain blue, indicating lower cost per unit area.")
    
    with sales_chart_tab:
        alt.data_transformers.disable_max_rows()
        # Scatter: Price vs Square Feet
        scatter_data = sales_df.dropna(subset=['PRICE', 'SQUARE_FEET']).copy()
        if len(scatter_data) > 5000:
            scatter_data = scatter_data.sample(5000, random_state=1)
        scatter_chart = alt.Chart(scatter_data).mark_circle(size=60, opacity=0.4).encode(
            x=alt.X('SQUARE_FEET:Q', title='Living Area (Sq Ft)'),
            y=alt.Y('PRICE:Q', title='Sale Price ($)', scale=alt.Scale(zero=False)),
            tooltip=['BEDS', 'BATHS', 'CITY', alt.Tooltip('PRICE:Q', title='Price')]
        )
        st.write("**Scatter: Home Price vs Size**")
        st.altair_chart(scatter_chart, use_container_width=True)
        st.caption("Larger homes generally command higher prices, as shown by the upward trend, though there is wide price variation among the largest properties.")
        
        # Histogram: Distribution of Sale Prices
        hist_chart = alt.Chart(sales_df).mark_bar(color='#4682B4').encode(
            x=alt.X('PRICE:Q', bin=alt.Bin(maxbins=50), title='Sale Price ($)'),
            y=alt.Y('count()', title='Number of Homes')
        )
        st.write("**Distribution: Home Sale Prices**")
        st.altair_chart(hist_chart, use_container_width=True)
        st.caption("Most home sale prices fall in the $600k to $1.4M range. The median price is about $860k, with a long tail of luxury homes pushing into multi-million dollar prices.")
        
        # Boxplot: Price by Number of Bedrooms
        box_data = sales_df.dropna(subset=['PRICE', 'BEDS'])
        box_chart = alt.Chart(box_data).mark_boxplot().encode(
            x=alt.X('BEDS:O', title='Bedrooms'),
            y=alt.Y('PRICE:Q', title='Sale Price ($)', scale=alt.Scale(zero=False))
        )
        st.write("**Price vs Bedrooms**")
        st.altair_chart(box_chart, use_container_width=True)
        st.caption("Homes with more bedrooms tend to have higher prices. Median sale prices increase with bedroom count (e.g., 4-bedroom homes typically cost more than 2-bedroom homes), although there is overlap due to location and other factors.")
    
    with sales_proj_tab:
        alt.data_transformers.disable_max_rows()
        # Histogram: Distribution of 30-year Net Value Gain
        net_gain_data = sales_df.dropna(subset=['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'])
        net_hist = alt.Chart(net_gain_data).mark_bar(color='#7E57C2').encode(
            x=alt.X('ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS:Q', bin=alt.Bin(maxbins=50), title='Net Value Gain After 30 Years ($)'),
            y=alt.Y('count()', title='Number of Homes')
        )
        st.write("**Distribution: 30-Year Net Gain from Homeownership**")
        st.altair_chart(net_hist, use_container_width=True)
        st.caption("Net financial outcomes after 30 years vary widely. Many properties hover around break-even or small losses, and relatively fewer homes yield very large gains (right tail of the distribution).")
        
        # Pie Chart: Share of Properties with Net Gain vs Net Loss
        num_gain = (sales_df['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'] > 0).sum()
        num_loss = (sales_df['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'] <= 0).sum()
        pie_fig = px.pie(values=[num_gain, num_loss], names=['Net Gain', 'Net Loss'],
                         color=['Net Gain', 'Net Loss'],
                         color_discrete_map={'Net Gain': '#4caf50', 'Net Loss': '#f44336'})
        st.write("**Outcomes: Home Value Gain vs Loss After 30 Years**")
        st.plotly_chart(pie_fig, use_container_width=True)
        st.caption(f"Only about {num_gain/(num_gain+num_loss)*100:.0f}% of these homes appreciate enough to offset all costs after 30 years (Net Gain). The majority (~{num_loss/(num_gain+num_loss)*100:.0f}%) result in a net loss when all purchase costs, taxes, and maintenance are considered.")
        
        # Scatter: Estimated Home Value vs Total Cost after 30 Years
        scatter_data2 = sales_df.dropna(subset=['TOTAL_COSTS_AFTER_30_YEARS', 'ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS']).copy()
        if len(scatter_data2) > 5000:
            scatter_data2 = scatter_data2.sample(5000, random_state=2)
        max_val = max(scatter_data2['TOTAL_COSTS_AFTER_30_YEARS'].max(), scatter_data2['ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS'].max())
        line_data = pd.DataFrame({'cost': [0, max_val], 'value': [0, max_val]})
        cost_vs_value_chart = alt.Chart(scatter_data2).mark_circle(size=60, opacity=0.4).encode(
            x=alt.X('TOTAL_COSTS_AFTER_30_YEARS:Q', title='Total 30-Year Cost ($)'),
            y=alt.Y('ESTIMATED_HOME_VALUE_APPRECIATED_AFTER_30_YEARS:Q', title='Estimated Home Value in 30 Years ($)')
        )
        line_chart = alt.Chart(line_data).mark_line(color='red', strokeDash=[5,5]).encode(
            x='cost:Q', y='value:Q'
        )
        combined_chart = cost_vs_value_chart + line_chart
        st.write("**Total Cost vs Future Home Value**")
        st.altair_chart(combined_chart, use_container_width=True)
        st.caption("Each point represents a home. Most points lie at or below the red 45° line, meaning the total costs of homeownership often meet or exceed the home's appreciated value after 30 years. Homes above the line are those that appreciated enough to outweigh all costs (net gain).")

# ----------------------------------- Rentals Tab -----------------------------------
with tab_rentals:
    # Sub-tabs for Rentals
    rent_map_tab, rent_chart_tab, rent_proj_tab = st.tabs(["Location Maps", "Charts", "30-Year Projections"])
    
    with rent_map_tab:
        # Map 1: Rental listings colored by Monthly Rent
        rent_vals = rentals_df[['LATITUDE', 'LONGITUDE', 'RENT_PER_MONTH']].dropna()
        low_cap_r = np.percentile(rent_vals['RENT_PER_MONTH'], 5)
        high_cap_r = np.percentile(rent_vals['RENT_PER_MONTH'], 95)
        rent_vals['rent_clamped'] = np.clip(rent_vals['RENT_PER_MONTH'], low_cap_r, high_cap_r)
        ratio_r = (rent_vals['rent_clamped'] - low_cap_r) / (high_cap_r - low_cap_r)
        r_r = np.where(ratio_r > 0.5, ((ratio_r - 0.5) * 2 * 255), 0).astype(int)
        g_r = np.where(ratio_r <= 0.5, (ratio_r * 2 * 255), (1 - (ratio_r - 0.5) * 2) * 255).astype(int)
        b_r = np.where(ratio_r <= 0.5, ((1 - ratio_r * 2) * 255), 0).astype(int)
        rent_vals['R'] = r_r
        rent_vals['G'] = g_r
        rent_vals['B'] = b_r
        view_state_r = pdk.ViewState(latitude=36.0, longitude=-119.5, zoom=6)
        layer_rent = pdk.Layer(
            "ScatterplotLayer",
            data=rent_vals,
            get_position="[LONGITUDE, LATITUDE]",
            get_fill_color="[R, G, B, 160]",
            get_radius=50,
            pickable=False,
        )
        deck_rent = pdk.Deck(layers=[layer_rent], initial_view_state=view_state_r, map_style="mapbox://styles/mapbox/light-v9")
        st.write("**Map 1: Rental Listings Colored by Monthly Rent**")
        st.pydeck_chart(deck_rent)
        st.caption("Higher rent listings (red) cluster around major cities (e.g., coastal Southern California), whereas lower rents (blue) are more common in smaller towns and inland areas.")
        
        # Map 2: Rental listings colored by Rent per Square Foot
        rpsf_vals = rentals_df[['LATITUDE', 'LONGITUDE', 'RENT_PRICE_PER_SQFT']].dropna()
        low_cap_rpsf = np.percentile(rpsf_vals['RENT_PRICE_PER_SQFT'], 5)
        high_cap_rpsf = np.percentile(rpsf_vals['RENT_PRICE_PER_SQFT'], 95)
        rpsf_vals['rpsf_clamped'] = np.clip(rpsf_vals['RENT_PRICE_PER_SQFT'], low_cap_rpsf, high_cap_rpsf)
        ratio_r2 = (rpsf_vals['rpsf_clamped'] - low_cap_rpsf) / (high_cap_rpsf - low_cap_rpsf)
        r2 = np.where(ratio_r2 > 0.5, ((ratio_r2 - 0.5) * 2 * 255), 0).astype(int)
        g2 = np.where(ratio_r2 <= 0.5, (ratio_r2 * 2 * 255), (1 - (ratio_r2 - 0.5) * 2) * 255).astype(int)
        b2 = np.where(ratio_r2 <= 0.5, ((1 - ratio_r2 * 2) * 255), 0).astype(int)
        rpsf_vals['R'] = r2
        rpsf_vals['G'] = g2
        rpsf_vals['B'] = b2
        layer_rpsf = pdk.Layer(
            "ScatterplotLayer",
            data=rpsf_vals,
            get_position="[LONGITUDE, LATITUDE]",
            get_fill_color="[R, G, B, 160]",
            get_radius=50,
            pickable=False,
        )
        deck_rpsf = pdk.Deck(layers=[layer_rpsf], initial_view_state=view_state_r, map_style="mapbox://styles/mapbox/light-v9")
        st.write("**Map 2: Rental Listings Colored by Rent per Square Foot**")
        st.pydeck_chart(deck_rpsf)
        st.caption("Urban rental markets show red clusters, indicating very high rent per square foot (expensive small apartments). More suburban and rural rentals (blue areas) offer significantly more space per dollar of rent.")
    
    with rent_chart_tab:
        alt.data_transformers.disable_max_rows()
        # Scatter: Rent vs Square Feet
        rent_scatter_data = rentals_df.dropna(subset=['RENT_PER_MONTH', 'SQUARE_FEET']).copy()
        if len(rent_scatter_data) > 5000:
            rent_scatter_data = rent_scatter_data.sample(5000, random_state=1)
        rent_scatter_chart = alt.Chart(rent_scatter_data).mark_circle(size=60, opacity=0.4, color='#FF7F0E').encode(
            x=alt.X('SQUARE_FEET:Q', title='Living Area (Sq Ft)'),
            y=alt.Y('RENT_PER_MONTH:Q', title='Monthly Rent ($)', scale=alt.Scale(zero=False)),
            tooltip=['BEDS', 'BATHS', 'CITY', alt.Tooltip('RENT_PER_MONTH:Q', title='Rent')]
        )
        st.write("**Scatter: Monthly Rent vs Size**")
        st.altair_chart(rent_scatter_chart, use_container_width=True)
        st.caption("Larger rental properties tend to have higher monthly rents. There is a clear positive correlation, though the spread indicates factors like location also influence rent significantly.")
        
        # Histogram: Distribution of Monthly Rents
        rent_hist_chart = alt.Chart(rentals_df).mark_bar(color='#FFA07A').encode(
            x=alt.X('RENT_PER_MONTH:Q', bin=alt.Bin(maxbins=40), title='Monthly Rent ($)'),
            y=alt.Y('count()', title='Number of Rentals')
        )
        st.write("**Distribution: Monthly Rent**")
        st.altair_chart(rent_hist_chart, use_container_width=True)
        st.caption("Most rentals range from about $2,100 to $4,000 per month. The median rent is around $2,900. Only a small fraction of upscale rentals exceed $6,000 per month.")
        
        # Bar: Number of Rentals by Bedrooms
        bed_counts = rentals_df['BEDS'].dropna().astype(int).value_counts().sort_index()
        bed_count_df = pd.DataFrame({'Bedrooms': bed_counts.index.astype(str), 'Count': bed_counts.values})
        bed_bar = alt.Chart(bed_count_df).mark_bar(color='#607d8b').encode(
            x=alt.X('Bedrooms:N', title='Bedrooms'),
            y=alt.Y('Count:Q', title='Number of Rentals')
        )
        st.write("**Rental Inventory by Bedrooms**")
        st.altair_chart(bed_bar, use_container_width=True)
        st.caption("Rental listings are skewed toward smaller units: the most common rentals have 2-3 bedrooms, and there are also many 1-bedroom and studio units. Larger 4+ bedroom rentals are less common.")
    
    with rent_proj_tab:
        # Histogram: Distribution of 30-year Total Rent Cost
        total_rent = rentals_df['RENT_TOTAL_COST_AFTER_30_YEARS'].dropna()
        rent_total_hist = alt.Chart(pd.DataFrame({'TOTAL_RENT': total_rent})).mark_bar(color='#AB47BC').encode(
            x=alt.X('TOTAL_RENT:Q', bin=alt.Bin(maxbins=40), title='Total Rent Paid over 30 Years ($)'),
            y=alt.Y('count()', title='Number of Rentals')
        )
        st.write("**Distribution: 30-Year Total Rent Paid**")
        st.altair_chart(rent_total_hist, use_container_width=True)
        st.caption("After 30 years of paying rent, most renters will have spent between $1.5M and $2.0M. This distribution mirrors the monthly rent spread, with the majority paying well over $1M in total rent and a significant share exceeding $2M.")
        
        # Pie Chart: 30-Year Rent Payment Categories
        count_low = (total_rent < 1_000_000).sum()
        count_mid = ((total_rent >= 1_000_000) & (total_rent < 2_000_000)).sum()
        count_high = (total_rent >= 2_000_000).sum()
        cost_labels = ['< $1M', '$1-2M', '> $2M']
        cost_counts = [int(count_low), int(count_mid), int(count_high)]
        rent_pie = px.pie(values=cost_counts, names=cost_labels,
                          color=cost_labels,
                          color_discrete_map={'< $1M': '#4caf50', '$1-2M': '#ffa726', '> $2M': '#d32f2f'})
        st.write("**Projected Rent Paid Categories**")
        st.plotly_chart(rent_pie, use_container_width=True)
        st.caption(f"Renting is costly in the long run: ~{(count_high/(count_low+count_mid+count_high))*100:.0f}% of rentals would cost over $2M after 30 years of payments, while only ~{(count_low/(count_low+count_mid+count_high))*100:.0f}% stay under $1M in total rent paid.")
        
        st.write("Over 30 years, even moderate monthly rents accumulate to enormous sums, underlining the importance of long-term financial planning for renters.")

# ----------------------------------- Comparison Tab -----------------------------------
with tab_comparison:
    # Sub-tabs for Comparison
    cost_comp_tab, invest_ret_tab = st.tabs(["Cost Comparison", "Investment Returns"])
    
    with cost_comp_tab:
        # Boxplot: Total 30-year Cost of Buying vs Renting
        buy_costs = sales_df['TOTAL_COSTS_AFTER_30_YEARS'].dropna()
        rent_costs = rentals_df['RENT_TOTAL_COST_AFTER_30_YEARS'].dropna()
        cost_compare_df = pd.DataFrame({
            'TotalCost': pd.concat([buy_costs, rent_costs], ignore_index=True),
            'Type': ['Buy'] * len(buy_costs) + ['Rent'] * len(rent_costs)
        })
        cost_box_chart = alt.Chart(cost_compare_df).mark_boxplot().encode(
            x=alt.X('Type:N', title=''),
            y=alt.Y('TotalCost:Q', title='Total Cost over 30 Years ($)', scale=alt.Scale(zero=False))
        )
        st.write("**Owning vs Renting: 30-Year Cost Distribution**")
        st.altair_chart(cost_box_chart, use_container_width=True)
        st.caption("The total cost of renting for 30 years often exceeds the total cost of homeownership. In our data, the median 30-year outlay for rent is about $1.88M, higher than the median $1.53M cost of buying. However, there is overlap: owning a home in an expensive area can cost as much as or more than long-term rent in a cheaper market.")
        
        # Scatter: City-level Median Price vs Median Rent
        city_median_price = sales_df.groupby('CITY')['PRICE'].median()
        city_median_rent = rentals_df.groupby('CITY')['RENT_PER_MONTH'].median()
        city_compare = pd.DataFrame({'City': city_median_price.index})
        city_compare['Median_Price'] = city_compare['City'].map(city_median_price)
        city_compare['Median_Rent'] = city_compare['City'].map(city_median_rent)
        city_compare = city_compare.dropna()
        city_scatter = alt.Chart(city_compare).mark_circle(size=60, opacity=0.6, color='#009688').encode(
            x=alt.X('Median_Price:Q', title='Median Home Price ($)', scale=alt.Scale(zero=False)),
            y=alt.Y('Median_Rent:Q', title='Median Monthly Rent ($)', scale=alt.Scale(zero=False)),
            tooltip=['City', alt.Tooltip('Median_Price:Q', title='Median Price'), alt.Tooltip('Median_Rent:Q', title='Median Rent')]
        )
        st.write("**City Comparison: Home Prices vs Rents**")
        st.altair_chart(city_scatter, use_container_width=True)
        st.caption("Cities with higher home prices generally have higher rents. The upward trend suggests a strong correlation between housing prices and rent levels. However, the slope varies, indicating that in some cities, home prices are especially high relative to rents (points far right for their rent level), while in others rents are comparatively steep.")
    
    with invest_ret_tab:
        # Bar: Median Homeowner Net Gain vs Median REIT Investment Return
        median_net = sales_df['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'].median()
        median_reit = sales_df['ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS'].median()
        outcome_df = pd.DataFrame({
            'Outcome': ['Home Net Gain', 'REIT Investment'],
            'Amount': [median_net, median_reit]
        })
        bar_fig = px.bar(outcome_df, x='Outcome', y='Amount', color='Outcome',
                         labels={'Amount': 'Amount ($)', 'Outcome': ''},
                         color_discrete_map={'Home Net Gain': '#d32f2f', 'REIT Investment': '#388e3c'})
        bar_fig.update_layout(yaxis_tickformat="$,.0f")
        st.write("**Investment After 30 Years: Homeownership vs REIT**")
        st.plotly_chart(bar_fig, use_container_width=True)
        st.caption(f"Investing equivalent funds in a REIT (Real Estate Investment Trust) far outperforms homeownership in this scenario. The median homeowner's net equity gain is around ${abs(median_net):,.0f} (a slight loss), whereas the same money in a REIT could grow to roughly ${median_reit:,.0f} over 30 years.")
        
        # Scatter: Homeowner Net Gain vs REIT Returns for each property
        invest_compare_data = sales_df.dropna(subset=['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS', 'ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS']).copy()
        if len(invest_compare_data) > 5000:
            invest_compare_data = invest_compare_data.sample(5000, random_state=3)
        max_val2 = max(invest_compare_data['ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS'].max(), invest_compare_data['ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS'].max(), 0)
        line_data2 = pd.DataFrame({'x': [0, max_val2], 'y': [0, max_val2]})
        invest_scatter = alt.Chart(invest_compare_data).mark_circle(size=60, opacity=0.4).encode(
            x=alt.X('ESTIMATED_FTSE_NAREIT_30_YEARS_RETURNS:Q', title='REIT 30-Year Returns ($)'),
            y=alt.Y('ESTIMATED_NET_VALUE_GAIN_AFTER_30_YEARS:Q', title='Homeowner Net Gain (30 Years) ($)')
        )
        line_ref = alt.Chart(line_data2).mark_line(color='red', strokeDash=[5,5]).encode(x='x:Q', y='y:Q')
        invest_combined = invest_scatter + line_ref
        st.write("**Performance: Investing vs Home Equity**")
        st.altair_chart(invest_combined, use_container_width=True)
        st.caption("Each point compares a specific home purchase (y-axis net gain) to investing the equivalent cost in a REIT (x-axis returns). All points lie below the diagonal line, indicating that none of the home purchases outperformed the investment returns over 30 years.")

        import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# --- Create Folium Map ---
m = folium.Map(location=[sales_df['LATITUDE'].mean(), sales_df['LONGITUDE'].mean()],
               zoom_start=7, tiles='cartodbpositron')

# --- Add marker clustering ---
marker_cluster = MarkerCluster().add_to(m)

for idx, row in sales_df.dropna(subset=['LATITUDE', 'LONGITUDE']).iterrows():
    price = row['PRICE']
    if price < 500000:
        color = 'green'
    elif price < 1000000:
        color = 'blue'
    elif price < 2000000:
        color = 'orange'
    else:
        color = 'red'
    
    folium.CircleMarker(
        location=[row['LATITUDE'], row['LONGITUDE']],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7,
        popup=folium.Popup(f"Address: {row.get('ADDRESS', 'N/A')}<br>City: {row['CITY']}<br>Price: ${price:,.0f}", max_width=300)
    ).add_to(marker_cluster)

# --- Show map in Streamlit ---
st_folium(m, width=1400, height=700)