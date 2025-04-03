#pipenv run streamlit run app.py
import re
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -- PAGE CONFIG & CUSTOM STYLING --
st.set_page_config(
    page_title="InfoCredit Dashboard",
    page_icon="💹",  # You can replace this emoji or remove it if not desired
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS to create a more pronounced blue theme and embed the logo.
st.markdown(
    f"""
    <style>
    /* Overall page background */
    .block-container {{
        background-color: #F3F6FA;
        padding-top: 20px !important;
        padding-bottom: 20px !important;
    }}
    /* Sidebar gradient background */
    .css-1cpxqw2, .css-1d391kg {{
        background: linear-gradient(180deg, #1E2C51 0%, #2c3d66 100%) !important;
    }}
    /* Make sidebar text white */
    .css-1cpxqw2, .css-1d391kg, .css-fblp2m {{
        color: #FFFFFF !important;
    }}
    /* Adjust titles/headings colors */
    h1, h2, h3, h4 {{
        color: #1E2C51; /* Dark navy text */
        margin-bottom: 0.5rem;
    }}
    /* Adjust normal text color */
    .css-10trblm, .st-ag, .st-bb, .st-ab {{
        color: #333 !important; 
    }}
    /* Buttons, text boxes, etc. */
    .stButton>button, .css-1cztu1l {{
        background-color: #1E2C51 !important; 
        color: #FFFFFF !important;
        border-radius: 5px !important;
    }}
    .stButton>button:hover {{
        background-color: #2c3d66 !important;
    }}
    /* Make table headers more distinctive (light grey with navy text) */
    .css-1jwm9li thead tr th {{
        background-color: #EFF2F6 !important;
        color: #1E2C51 !important;
    }}
    /* Scrollbar styling (optional) */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: #E6E9ED;
    }}
    ::-webkit-scrollbar-thumb {{
        background-color: #2c3d66;
        border-radius: 6px;
        border: 2px solid #E6E9ED;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache
def load_data():
    # Adjust file paths as needed
    df0 = pd.read_csv("cs.csv")  # Includes INCORP_DATE and ALIAS_NAME
    df1 = pd.read_csv("file1.csv")
    df2 = pd.read_csv("file2.csv")
    df7 = pd.read_csv("file7.csv")
    return df0, df1, df2, df7

def data(df0, df1, df2, df7):
    merged_12 = pd.merge(df1, df2, on="National.ID", how="left")
    merged_all = pd.merge(merged_12, df7, on="National.ID", how="left")
    final_merged = pd.merge(
        merged_all,
        df0[['ALIAS_NAME', 'INCORP_DATE']], 
        left_on="National.ID", 
        right_on="ALIAS_NAME", 
        how="left"
    )
    return final_merged

def parse_pkd_list(pkd_text):
    """
    Takes a string of comma-separated PKD codes (each up to 4 digits).
    Returns a list of valid PKD codes (strings).
    """
    if not pkd_text.strip():
        return []
    raw_pkds = [pkd.strip() for pkd in pkd_text.split(',')]
    valid_pkds = []
    for code in raw_pkds:
        if re.match(r'^\d{1,4}$', code):
            valid_pkds.append(code)
        else:
            st.warning(f"Invalid PKD code: {code} (ignored)")
    return valid_pkds

def parse_nip_list(nip_text):
    if not nip_text.strip():
        return []
    nips = [n.strip() for n in nip_text.split(',')]
    return nips

def filter_by_multiple_pkd(df, pkd_list):
    if "Main.industry.code" not in df.columns:
        st.warning("No 'Main.industry.code' in data. Skipping PKD filter.")
        return df, True
    
    condition = df["Main.industry.code"].astype(str).apply(
        lambda x: any(x.startswith(code) for code in pkd_list)
    )
    filtered = df[condition]
    found_any = not filtered.empty
    return filtered, found_any

def filter_by_nip(df, nip_list):
    if "NIP" not in df.columns:
        st.warning("No 'NIP' column in data. Skipping NIP filter.")
        return df, True
    filtered = df[df["NIP"].astype(str).isin(nip_list)]
    found_any = not filtered.empty
    return filtered, found_any

def calculate_ratios(df):
    """
    Existing ratio calculations, plus EBIT Margin.
    """
    # EBIT
    if {'Operating.revenue', 'Costs.of.goods.sold', 'Other.operating.expenses'}.issubset(df.columns):
        df['EBIT'] = df['Operating.revenue'] - df['Costs.of.goods.sold'] - df['Other.operating.expenses']
    else:
        df['EBIT'] = None

    df['LATEST_TURNOVER'] = df.get('Sales', None)
    df['PRE_TAX_PROFIT'] = df.get('P.L.before.taxation', None)
    df['PROFIT'] = df.get('P.L.after.taxation', None)
    df['EMPLOYEES'] = df.get('Number.of.employees', None)
    df['TOTAL_FIXED_ASSETS'] = df.get('Fixed.assets', None)
    df['TOTAL_CURRENT_ASSETS'] = df.get('Current.assets', None)

    def compute_total_curr_liab(row):
        if 'Accounting.practice' not in row or pd.isna(row['Accounting.practice']):
            return None
        if row['Accounting.practice'] == 0:
            return row['debtors'] + row['creditors'] + row['other.current.liabilities']
        else:
            return row['loans'] + row['creditors']

    if 'Accounting.practice' in df.columns:
        df['TOTAL_CURRENT_LIABILITIES'] = df.apply(compute_total_curr_liab, axis=1)
    else:
        df['TOTAL_CURRENT_LIABILITIES'] = None

    df['TOTAL_LONG_TERM_LIABILITIES'] = df.get('Non.current.liabilities', None)
    df['SHAREHOLDERS_FUNDS'] = df.get('Shareholders.Funds', None)
    df['NET_WORTH'] = df.get('Shareholders.Funds', 0) - df.get('capital', 0)

    def compute_working_capital(row):
        if 'Accounting.practice' not in row or pd.isna(row['Accounting.practice']):
            return None
        if row['Accounting.practice'] == 0:
            return row['Current.assets'] - (row['debtors'] + row['creditors'] + row['other.current.liabilities'])
        else:
            return row['Current.assets'] - (row['loans'] + row['creditors'])

    if 'Current.assets' in df.columns:
        df['WORKING_CAPITAL'] = df.apply(compute_working_capital, axis=1)
    else:
        df['WORKING_CAPITAL'] = None

    def compute_current_ratio(row):
        if 'Accounting.practice' not in row or pd.isna(row['Accounting.practice']):
            return None
        if row['Accounting.practice'] == 0:
            denominator = row['debtors'] + row['creditors'] + row['other.current.liabilities']
        else:
            denominator = row['loans'] + row['creditors']
        if denominator == 0 or pd.isna(denominator):
            return None
        return row['Current.assets'] / denominator if pd.notna(row['Current.assets']) else None

    if 'Current.assets' in df.columns:
        df['CURRENT_RATIO'] = df.apply(compute_current_ratio, axis=1)
    else:
        df['CURRENT_RATIO'] = None

    # Additional ratio examples
    if 'Sales' in df.columns and (df['Sales'] != 0).any():
        df['PRE_TAX_PROFIT_MARGIN'] = (df['P.L.before.taxation'] * 100) / df['Sales']
    else:
        df['PRE_TAX_PROFIT_MARGIN'] = None

    if 'Total.assets' in df.columns and (df['Total.assets'] != 0).any():
        df['RETURN_ON_TOTAL_ASSETS_EMPLOYED'] = (df['P.L.before.taxation'] * 100) / df['Total.assets']
    else:
        df['RETURN_ON_TOTAL_ASSETS_EMPLOYED'] = None

    def compute_total_debt_ratio(row):
        if 'Accounting.practice' not in row or pd.isna(row['Accounting.practice']):
            return None
        if row['Accounting.practice'] == 0:
            numerator = row['Non.current.liabilities'] + (
                row['debtors'] + row['creditors'] + row['other.current.liabilities']
            )
        else:
            numerator = row['Non.current.liabilities'] + (row['loans'] + row['creditors'])
        sf = row['Shareholders.Funds']
        if sf == 0 or pd.isna(sf):
            return None
        return numerator / sf

    df['TOTAL_DEBT_RATIO'] = df.apply(compute_total_debt_ratio, axis=1)

    def compute_current_debt_ratio(row):
        if 'Accounting.practice' not in row or pd.isna(row['Accounting.practice']):
            return None
        if row['Accounting.practice'] == 0:
            numerator = row['debtors'] + row['creditors'] + row['other.current.liabilities']
        else:
            numerator = row['loans'] + row['creditors']
        sf = row['Shareholders.Funds']
        if sf == 0 or pd.isna(sf):
            return None
        return numerator / sf

    df['CURRENT_DEBT_RATIO'] = df.apply(compute_current_debt_ratio, axis=1)

    # Quick Ratio
    def compute_quick_ratio(row):
        if pd.isna(row.get('TOTAL_CURRENT_LIABILITIES', None)) or row['TOTAL_CURRENT_LIABILITIES'] == 0:
            return None
        current_assets = row.get('Current.assets', None)
        if pd.isna(current_assets):
            return None
        inventories = row.get('Inventories', 0)
        prepayments = row.get('Prepayments', 0)
        if pd.isna(inventories):
            inventories = 0
        if pd.isna(prepayments):
            prepayments = 0
        quick_assets = current_assets - inventories - prepayments
        return quick_assets / row['TOTAL_CURRENT_LIABILITIES']

    df['QUICK_RATIO'] = df.apply(compute_quick_ratio, axis=1)

    # Receivables Turnover = (debtors / Sales) * 365
    def compute_receivables_turnover(row):
        receivables = row.get('debtors', None)
        sales = row.get('Sales', None)
        if pd.isna(receivables) or pd.isna(sales) or sales == 0:
            return None
        return (receivables / sales) * 365

    df['RECEIVABLES_TURNOVER'] = df.apply(compute_receivables_turnover, axis=1)

    # Payables Turnover = (creditors / Sales) * 365
    def compute_payables_turnover(row):
        payables = row.get('creditors', None)
        sales = row.get('Sales', None)
        if pd.isna(payables) or pd.isna(sales) or sales == 0:
            return None
        return (payables / sales) * 365

    df['PAYABLES_TURNOVER'] = df.apply(compute_payables_turnover, axis=1)

    # EBIT Margin
    def compute_ebit_margin(row):
        ebit_val = row.get('EBIT', None)
        fin_revenue = row.get('Financial.revenue', None)
        if pd.isna(ebit_val) or pd.isna(fin_revenue) or fin_revenue == 0:
            return None
        return (ebit_val / fin_revenue) * 100

    df['EBIT_MARGIN'] = df.apply(compute_ebit_margin, axis=1)

    return df

def compute_yoy_changes(df):
    """
    Groups data by (National.ID, year) and calculates sums for revenue, employees, EBIT, profit;
    then calculates yoy changes for multiple metrics, plus yoy changes in:
      - Net Margin
      - EBIT Margin
    """
    if 'Closing.date.of.statement' not in df.columns:
        return df  # skip yoy if no date

    # Extract year from YYYYMMDD
    df['year'] = (df['Closing.date.of.statement'] // 10000).astype(int)

    # We'll sum for revenue, employees, EBIT, PROFIT. We'll also store them for margin calculations.
    agg_dict = {
        'Financial.revenue': 'sum',
        'Number.of.employees': 'sum',
        'EBIT': 'sum',
        'PROFIT': 'sum'
    }

    has_avg_salary = False
    if 'Average_salary' in df.columns:
        agg_dict['Average_salary'] = 'mean'
        has_avg_salary = True

    group = df.groupby(['National.ID', 'year'], as_index=False).agg(agg_dict).rename(columns={
        'Financial.revenue': 'AnnualRevenue',
        'Number.of.employees': 'AnnualEmployees',
        'EBIT': 'AnnualEBIT',
        'PROFIT': 'AnnualProfit',
        'Average_salary': 'AnnualAvgSalary' if has_avg_salary else None
    })

    # Also compute NetMargin, EbitMargin at the annual level for yoy
    # NetMargin = (AnnualProfit / AnnualRevenue) * 100
    # EbitMargin = (AnnualEBIT / AnnualRevenue) * 100
    group['AnnualNetMargin'] = group.apply(
        lambda row: (row['AnnualProfit'] / row['AnnualRevenue']) * 100
        if (pd.notna(row['AnnualProfit']) and pd.notna(row['AnnualRevenue']) and row['AnnualRevenue'] != 0)
        else None,
        axis=1
    )
    group['AnnualEbitMargin'] = group.apply(
        lambda row: (row['AnnualEBIT'] / row['AnnualRevenue']) * 100
        if (pd.notna(row['AnnualEBIT']) and pd.notna(row['AnnualRevenue']) and row['AnnualRevenue'] != 0)
        else None,
        axis=1
    )

    # Sort
    group.sort_values(by=['National.ID', 'year'], inplace=True)

    # SHIFT previous year for yoy
    # 1) Revenue yoy
    group['PrevRevenue'] = group.groupby('National.ID')['AnnualRevenue'].shift(1)
    group['REVENUE_GROWTH_NOMINAL'] = group['AnnualRevenue'] - group['PrevRevenue']
    group['REVENUE_GROWTH_PERCENT'] = group.apply(
        lambda row: (row['REVENUE_GROWTH_NOMINAL'] / row['PrevRevenue'] * 100)
        if (pd.notna(row['REVENUE_GROWTH_NOMINAL']) and row['PrevRevenue'] != 0) else None,
        axis=1
    )

    # 2) Employees yoy
    group['PrevEmployees'] = group.groupby('National.ID')['AnnualEmployees'].shift(1)
    group['EMPLOYEES_GROWTH_NOMINAL'] = group['AnnualEmployees'] - group['PrevEmployees']
    group['EMPLOYEES_GROWTH_PERCENT'] = group.apply(
        lambda row: (row['EMPLOYEES_GROWTH_NOMINAL'] / row['PrevEmployees'] * 100)
        if (pd.notna(row['EMPLOYEES_GROWTH_NOMINAL']) and row['PrevEmployees'] != 0) else None,
        axis=1
    )

    # 3) EBIT yoy
    group['PrevEBIT'] = group.groupby('National.ID')['AnnualEBIT'].shift(1)
    group['EBIT_GROWTH_NOMINAL'] = group['AnnualEBIT'] - group['PrevEBIT']
    group['EBIT_GROWTH_PERCENT'] = group.apply(
        lambda row: (row['EBIT_GROWTH_NOMINAL'] / row['PrevEBIT'] * 100)
        if (pd.notna(row['EBIT_GROWTH_NOMINAL']) and row['PrevEBIT'] != 0) else None,
        axis=1
    )

    # 4) Profit yoy
    group['PrevProfit'] = group.groupby('National.ID')['AnnualProfit'].shift(1)
    group['PROFIT_GROWTH_NOMINAL'] = group['AnnualProfit'] - group['PrevProfit']
    group['PROFIT_GROWTH_PERCENT'] = group.apply(
        lambda row: (row['PROFIT_GROWTH_NOMINAL'] / row['PrevProfit'] * 100)
        if (pd.notna(row['PROFIT_GROWTH_NOMINAL']) and row['PrevProfit'] != 0) else None,
        axis=1
    )

    # 5) Avg Salary yoy
    if has_avg_salary:
        group['PrevAvgSalary'] = group.groupby('National.ID')['AnnualAvgSalary'].shift(1)
        group['AVG_SALARY_GROWTH_NOMINAL'] = group['AnnualAvgSalary'] - group['PrevAvgSalary']
        group['AVG_SALARY_GROWTH_PERCENT'] = group.apply(
            lambda row: (row['AVG_SALARY_GROWTH_NOMINAL'] / row['PrevAvgSalary'] * 100)
            if (pd.notna(row['AVG_SALARY_GROWTH_NOMINAL']) and row['PrevAvgSalary'] != 0) else None,
            axis=1
        )

    # 6) Net Margin yoy
    group['PrevNetMargin'] = group.groupby('National.ID')['AnnualNetMargin'].shift(1)
    group['NET_MARGIN_YOY_NOMINAL'] = group['AnnualNetMargin'] - group['PrevNetMargin']
    group['NET_MARGIN_YOY_PERCENT'] = group.apply(
        lambda row: (row['NET_MARGIN_YOY_NOMINAL'] / row['PrevNetMargin'] * 100)
        if (pd.notna(row['NET_MARGIN_YOY_NOMINAL']) and row['PrevNetMargin'] != 0) else None,
        axis=1
    )

    # 7) EBIT Margin yoy
    group['PrevEbitMargin'] = group.groupby('National.ID')['AnnualEbitMargin'].shift(1)
    group['EBIT_MARGIN_YOY_NOMINAL'] = group['AnnualEbitMargin'] - group['PrevEbitMargin']
    group['EBIT_MARGIN_YOY_PERCENT'] = group.apply(
        lambda row: (row['EBIT_MARGIN_YOY_NOMINAL'] / row['PrevEbitMargin'] * 100)
        if (pd.notna(row['EBIT_MARGIN_YOY_NOMINAL']) and row['PrevEbitMargin'] != 0) else None,
        axis=1
    )

    yoy_cols = [
        'National.ID', 'year',
        'REVENUE_GROWTH_NOMINAL', 'REVENUE_GROWTH_PERCENT',
        'EMPLOYEES_GROWTH_NOMINAL', 'EMPLOYEES_GROWTH_PERCENT',
        'EBIT_GROWTH_NOMINAL', 'EBIT_GROWTH_PERCENT',
        'PROFIT_GROWTH_NOMINAL', 'PROFIT_GROWTH_PERCENT',
        'NET_MARGIN_YOY_NOMINAL', 'NET_MARGIN_YOY_PERCENT',
        'EBIT_MARGIN_YOY_NOMINAL', 'EBIT_MARGIN_YOY_PERCENT'
    ]
    if has_avg_salary:
        yoy_cols += ['AVG_SALARY_GROWTH_NOMINAL', 'AVG_SALARY_GROWTH_PERCENT']

    df_merged = pd.merge(df, group[yoy_cols], how='left', on=['National.ID', 'year'])
    return df_merged

def main():
    # Display the InfoCredit logo plus a custom title area
    st.markdown(
        """
        <div style='display:flex; align-items:center; margin-bottom:20px;'>
            <img src="https://infocredit.pl/wp-content/uploads/2024/07/c4c14dfe6fd5a71e404b2f6eab4a3f14.svg"
                 alt="InfoCredit Logo"
                 style='height:60px; margin-right:20px;' />
            <h1 style='margin:0;'>Sector Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load & merge data
    df0, df1, df2, df7 = load_data()
    overview_data = data(df0, df1, df2, df7)  # Merged DataFrame

    # Sidebar inputs
    st.sidebar.header("Filters")

    # Name filter (at the beginning of the sidebar)
    name_input = st.sidebar.text_input("Filter by Name:", "").strip()
    if name_input:
        name_input = name_input.lower()

    # Apply Name Filter
    if name_input:
        if 'Name' in overview_data.columns:
            # Case-insensitive partial match for the input string
            overview_data = overview_data[overview_data['Name'].str.lower().str.contains(name_input, na=False)]
        else:
            st.warning("No 'Name' column found in data. Skipping name filter.")

    # PKD codes
    pkd_text = st.sidebar.text_area("Enter PKD Codes (comma-separated):", "")
    pkd_list = parse_pkd_list(pkd_text)

    # NIPs
    nip_text = st.sidebar.text_area("Enter NIPs (comma-separated):", "")
    nip_list = parse_nip_list(nip_text)

    # Year and number of years
    year_input = st.sidebar.text_input("Enter the starting year (leave blank for all years):", "")
    num_years = st.sidebar.number_input("Enter the number of years to show:", min_value=1, max_value=100, value=1)

    # Convert year_input to integer if provided
    if year_input.strip():
        try:
            year_input = int(year_input)
        except ValueError:
            st.sidebar.error("Please enter a valid starting year (e.g., 2021).")
            return
    else:
        year_input = None

    # Country, province, city
    country_input = st.sidebar.text_input("Enter Country Code (e.g., 'PL'):", "")
    show_province_city = st.sidebar.checkbox("Show Province and City Filters")
    province_input = ""
    city_input = ""

    if show_province_city:
        if 'Region' in overview_data.columns:
            province_options = ["All Regions"] + list(overview_data['Region'].dropna().unique())
            province_input = st.sidebar.selectbox("Select Province Code:", options=province_options)
        else:
            st.warning("No 'Region' column found in data.")

        if 'City' in overview_data.columns:
            city_options = ["All Cities"] + list(overview_data['City'].dropna().unique())
            city_input = st.sidebar.multiselect("Select City Name:", options=city_options)
        else:
            st.warning("No 'City' column found in data.")

    # Start filtering
    filtered_df = overview_data.copy()

    # PKD filter
    if pkd_list:
        filtered_df, found_pkd = filter_by_multiple_pkd(filtered_df, pkd_list)
        if not found_pkd:
            st.warning(f"No rows found for PKD codes: {', '.join(pkd_list)}")

    # NIP filter
    if nip_list:
        filtered_df, found_nip = filter_by_nip(filtered_df, nip_list)
        if not found_nip:
            st.warning(f"No rows found for given NIPs: {', '.join(nip_list)}")

    # Year filter
    if year_input:
        if 'Closing.date.of.statement' in filtered_df.columns:
            start_date = int(f"{year_input}0101")
            end_date = int(f"{year_input + num_years - 1}1231")
            filtered_df = filtered_df[
                (filtered_df['Closing.date.of.statement'] >= start_date) &
                (filtered_df['Closing.date.of.statement'] <= end_date)
            ]
        else:
            st.warning("No 'Closing.date.of.statement' column in data. Skipping date filter.")

    # Country filter
    if country_input:
        if 'Country' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Country'] == country_input]
        else:
            st.warning("No 'Country' column in data. Skipping country filter.")

    # Province filter
    if province_input and province_input != "All Regions" and 'Region' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Region'] == province_input]

    # City filter
    if city_input and "All Cities" not in city_input and 'City' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['City'].isin(city_input)]

    # If no data left, stop
    if filtered_df.empty:
        st.error("No data left after filtering.")
        return

    # Ensure National ID is string
    filtered_df['National.ID'] = filtered_df['National.ID'].astype(str)

    # Ratios
    filtered_df_with_ratios = calculate_ratios(filtered_df)

    # If multi-year analysis, do YOY
    if year_input and num_years > 1:
        filtered_df_with_ratios = compute_yoy_changes(filtered_df_with_ratios)

    # Display data
    st.subheader("Filtered Data Overview")

    st.write(f"Number of rows: {len(filtered_df_with_ratios)}")
    st.write(f"Number of unique companies (National.ID): {filtered_df_with_ratios['National.ID'].nunique()}")


        # Convert 'Main.industry.code' to whole number if it exists
    if 'Main.industry.code' in filtered_df_with_ratios.columns:
        filtered_df_with_ratios['Main.industry.code'] = filtered_df_with_ratios['Main.industry.code'].fillna(0).astype(str)
    if 'INCORP_DATE' in filtered_df_with_ratios.columns:
        filtered_df_with_ratios['INCORP_DATE'] = filtered_df_with_ratios['INCORP_DATE'].astype(str)
    
    st.dataframe(filtered_df_with_ratios.head(1000))

   
    # Financial Ratios Table
    st.subheader("Financial Ratios Table")
    ratios_columns = [
        'Name',
        'Main.industry.code',
        'INCORP_DATE', 
        'National.ID',
        'Financial.revenue',
        'Number.of.employees',
        'EBIT',
        'EBIT_MARGIN',
        'PROFIT',
        'CURRENT_RATIO',
        'QUICK_RATIO',
        'RECEIVABLES_TURNOVER',
        'PAYABLES_TURNOVER',
        'CURRENT_DEBT_RATIO',
        'TOTAL_DEBT_RATIO',
        # YOY columns (if multi-year)
        'REVENUE_GROWTH_NOMINAL','REVENUE_GROWTH_PERCENT',
        'EMPLOYEES_GROWTH_NOMINAL','EMPLOYEES_GROWTH_PERCENT',
        'EBIT_GROWTH_NOMINAL','EBIT_GROWTH_PERCENT',
        'PROFIT_GROWTH_NOMINAL','PROFIT_GROWTH_PERCENT',
        'NET_MARGIN_YOY_NOMINAL','NET_MARGIN_YOY_PERCENT',
        'EBIT_MARGIN_YOY_NOMINAL','EBIT_MARGIN_YOY_PERCENT'
    ]

    valid_ratio_cols = [col for col in ratios_columns if col in filtered_df_with_ratios.columns]

    # Convert 'Main.industry.code' to whole number if it exists
    if 'Main.industry.code' in filtered_df_with_ratios.columns:
        filtered_df_with_ratios['Main.industry.code'] = filtered_df_with_ratios['Main.industry.code'].fillna(0).astype(str)

    if 'INCORP_DATE' in filtered_df_with_ratios.columns:
        filtered_df_with_ratios['INCORP_DATE'] = filtered_df_with_ratios['INCORP_DATE'].astype(str)


    ratios_table = filtered_df_with_ratios[valid_ratio_cols]
    st.dataframe(ratios_table.head(1000))

    # Summary Metrics
    st.subheader("Summary Metrics")

    # Basic counts
    if 'Financial.revenue' in filtered_df_with_ratios.columns:
        num_companies_with_financial_data = filtered_df_with_ratios['Financial.revenue'].notna().sum()
        num_companies_without_financial_data = filtered_df_with_ratios['Financial.revenue'].isna().sum()
    else:
        num_companies_with_financial_data = 0
        num_companies_without_financial_data = len(filtered_df_with_ratios)
    st.write(f"Number of companies with financial data: {num_companies_with_financial_data}")
    st.write(f"Number of companies without financial data: {num_companies_without_financial_data}")

    # Profit quartiles
    if 'PROFIT' in filtered_df_with_ratios.columns:
        profit_quartiles = filtered_df_with_ratios['PROFIT'].quantile([0.25, 0.5, 0.75]).to_dict()
        st.write("Profit Quartiles (in filtered data):")
        st.write(f"  - Q1 (25th percentile): {profit_quartiles.get(0.25, 'N/A')}")
        st.write(f"  - Median (50th percentile): {profit_quartiles.get(0.5, 'N/A')}")
        st.write(f"  - Q3 (75th percentile): {profit_quartiles.get(0.75, 'N/A')}")

    # EBIT Margin Quartiles
    if 'EBIT_MARGIN' in filtered_df_with_ratios.columns:
        margin_quartiles = filtered_df_with_ratios['EBIT_MARGIN'].dropna().quantile([0.25, 0.5, 0.75]).to_dict()
        st.write("EBIT Margin Quartiles (%):")
        st.write(f"  - Q1 (25th percentile): {margin_quartiles.get(0.25, 'N/A'):.2f}%")
        st.write(f"  - Median (50th percentile): {margin_quartiles.get(0.5, 'N/A'):.2f}%")
        st.write(f"  - Q3 (75th percentile): {margin_quartiles.get(0.75, 'N/A'):.2f}%")

    # Industry aggregates
    st.markdown("### Industry-Wide Aggregates")
    if 'Financial.revenue' in filtered_df_with_ratios.columns:
        total_revenue = filtered_df_with_ratios['Financial.revenue'].fillna(0).sum()
        st.write(f"**Total Revenue (Industry)**: {total_revenue:,.2f}")
    if 'Number.of.employees' in filtered_df_with_ratios.columns:
        total_employees = filtered_df_with_ratios['Number.of.employees'].fillna(0).sum()
        st.write(f"**Total Employees**: {total_employees:,}")
    if 'Total.assets' in filtered_df_with_ratios.columns:
        total_assets = filtered_df_with_ratios.get('Total.assets', pd.Series()).fillna(0).sum()
        st.write(f"**Sum of Total Assets**: {total_assets:,.2f}")
    if 'Shareholders.Funds' in filtered_df_with_ratios.columns:
        total_equity = filtered_df_with_ratios['Shareholders.Funds'].fillna(0).sum()
        st.write(f"**Sum of Shareholders' Funds (Equity)**: {total_equity:,.2f}")
    if 'loans' in filtered_df_with_ratios.columns:
        total_loans = filtered_df_with_ratios['loans'].fillna(0).sum()
        st.write(f"**Sum of Long-term Loans**: {total_loans:,.2f}")
    if 'TOTAL_CURRENT_LIABILITIES' in filtered_df_with_ratios.columns:
        total_st_liab = filtered_df_with_ratios['TOTAL_CURRENT_LIABILITIES'].fillna(0).sum()
        st.write(f"**Sum of Short-term Liabilities**: {total_st_liab:,.2f}")
    if 'EBIT' in filtered_df_with_ratios.columns:
        total_ebit = filtered_df_with_ratios['EBIT'].fillna(0).sum()
        st.write(f"**Sum of EBIT**: {total_ebit:,.2f}")
    if 'PROFIT' in filtered_df_with_ratios.columns:
        total_profit = filtered_df_with_ratios['PROFIT'].fillna(0).sum()
        st.write(f"**Sum of Net Profit**: {total_profit:,.2f}")

    st.sidebar.success("Filtering & basic metrics complete.")

    # ===========================
    # NEW: BENCHMARKING FEATURE
    # ===========================
    st.subheader("Benchmarking Feature")

    # Ensure 'National.ID' or 'ALIAS_NAME' exists for selection
    if 'Name' not in filtered_df_with_ratios.columns:
        st.error("'Name' column is missing from the data.")
    else:
        # 1. Select Company to Benchmark
        st.markdown("### Select Company for Benchmarking")
        company_options = filtered_df_with_ratios['Name'].unique()
        selected_company = st.selectbox("Name:", options=company_options)

        # Retrieve selected company's data
        selected_company_data = filtered_df_with_ratios[filtered_df_with_ratios['Name'] == selected_company]

        if selected_company_data.empty:
            st.error("Selected company data is not available.")
        else:
            # 2. Select Variables for Benchmarking
            st.markdown("### Select Variables to Benchmark")
            # Define a list of important variables for benchmarking
            # You can customize this list based on your data
            important_variables = [
            'Financial.revenue',
            'EBIT',
            'PROFIT',
            'CURRENT_RATIO',
            'QUICK_RATIO',
            'RECEIVABLES_TURNOVER',
            'PAYABLES_TURNOVER',
            'EBIT_MARGIN',
            'TOTAL_DEBT_RATIO',
            'CURRENT_DEBT_RATIO',
            'PRE_TAX_PROFIT_MARGIN',
            'RETURN_ON_TOTAL_ASSETS_EMPLOYED',
            # Additional variables from file7
            'Gross.profit',
            'Operating.P.L',
            'Financial.P.L',
            'Cash_flow',
            'Costs.of.employees',
            'Depreciation',
            'Interest.paid',
            'Export.turnover',
            'Material.costs'
        ]
            # Filter available variables based on DataFrame columns
            available_variables = [var for var in important_variables if var in filtered_df_with_ratios.columns]

            selected_variables = st.multiselect(
                "Select one or more variables for benchmarking:",
                options=available_variables
            )

            # Proceed only if at least one variable is selected
            if selected_variables:
                if len(selected_variables) == 1:
                    # Boxplot Comparison
                    variable = selected_variables[0]

                    # Prepare data for boxplot
                    boxplot_data = filtered_df_with_ratios[variable].dropna()

                    # Get the selected company's value for the variable
                    company_value = selected_company_data[variable].values[0]

                    # Create a boxplot using Plotly
                    fig_box = px.box(
                        y=boxplot_data,
                        title=f"Comparison of {variable} for Company {selected_company}",
                        labels={variable: variable},
                        points="all"
                    )

                    # Add a scatter point for the selected company
                    fig_box.add_trace(
                        go.Scatter(
                            x=[0],
                            y=[company_value],
                            mode='markers',
                            marker=dict(color='red', size=12, symbol='diamond'),
                            name=f'Company {selected_company}'
                        )
                    )

                    # Update layout to remove x-axis ticks
                    fig_box.update_layout(xaxis=dict(showticklabels=False))

                    st.plotly_chart(fig_box, use_container_width=True)

                else:
                    # Radar Chart Comparison
                    # Calculate average values for selected variables
                    avg_values = filtered_df_with_ratios[selected_variables].mean().values.tolist()

                    # Get the selected company's values for the variables
                    company_values = selected_company_data[selected_variables].iloc[0].values.tolist()

                    # Radar charts require the first value to be repeated at the end to close the circle
                    categories = selected_variables + [selected_variables[0]]
                    company_values += [company_values[0]]
                    avg_values += [avg_values[0]]

                    # Create Radar Chart using Plotly
                    fig_radar = go.Figure()

                    # Add company trace
                    fig_radar.add_trace(go.Scatterpolar(
                        r=company_values,
                        theta=categories,
                        fill='toself',
                        name=f'Company {selected_company}'
                    ))

                    # Add average trace
                    fig_radar.add_trace(go.Scatterpolar(
                        r=avg_values,
                        theta=categories,
                        fill='toself',
                        name='Industry Average'
                    ))

                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                # Optionally set range or other properties
                            )
                        ),
                        title=f"Radar Chart Comparison for Company {selected_company}",
                        showlegend=True
                    )

                    st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info("Please select at least one variable to benchmark.")


    # ===============
    # RANKINGS
    # ===============
    st.subheader("Rankings")

    ranking_limit = st.selectbox("Select Top X companies to display:", [5, 10, 50, 100], index=0)

    # Define available ranking categories
    ranking_categories = {
        "Top X by Revenue": {
            "column": "Financial.revenue",
            "ascending": False,
            "columns_to_display": ["Name", "Financial.revenue"]
        },
        "Top X by Revenue Growth (%)": {
            "column": "REVENUE_GROWTH_PERCENT",
            "ascending": False,
            "columns_to_display": ["Name", "REVENUE_GROWTH_PERCENT"]
        },
        "Top X by Revenue Growth (Nominal)": {
            "column": "REVENUE_GROWTH_NOMINAL",
            "ascending": False,
            "columns_to_display": ["Name", "REVENUE_GROWTH_NOMINAL"]
        },
        "Top X by Employees": {
            "column": "Number.of.employees",
            "ascending": False,
            "columns_to_display": ["Name", "Number.of.employees"]
        },
        "Top X by Employees Growth (%)": {
            "column": "EMPLOYEES_GROWTH_PERCENT",
            "ascending": False,
            "columns_to_display": ["Name", "EMPLOYEES_GROWTH_PERCENT"]
        },
        "Top X by Employees Growth (Nominal)": {
            "column": "EMPLOYEES_GROWTH_NOMINAL",
            "ascending": False,
            "columns_to_display": ["Name", "EMPLOYEES_GROWTH_NOMINAL"]
        },
        "Top X by EBIT": {
            "column": "EBIT",
            "ascending": False,
            "columns_to_display": ["Name", "EBIT"]
        },
        "Top X by EBIT Growth (%)": {
            "column": "EBIT_GROWTH_PERCENT",
            "ascending": False,
            "columns_to_display": ["Name", "EBIT_GROWTH_PERCENT"]
        },
        "Top X by EBIT Growth (Nominal)": {
            "column": "EBIT_GROWTH_NOMINAL",
            "ascending": False,
            "columns_to_display": ["Name", "EBIT_GROWTH_NOMINAL"]
        },
        "Top X by Net Profit": {
            "column": "PROFIT",
            "ascending": False,
            "columns_to_display": ["Name", "PROFIT"]
        },
        "Top X by Profit Growth (%)": {
            "column": "PROFIT_GROWTH_PERCENT",
            "ascending": False,
            "columns_to_display": ["Name", "PROFIT_GROWTH_PERCENT"]
        },
        "Top X by Profit Growth (Nominal)": {
            "column": "PROFIT_GROWTH_NOMINAL",
            "ascending": False,
            "columns_to_display": ["Name", "PROFIT_GROWTH_NOMINAL"]
        },
    }

    # Multiselect for ranking categories
    selected_rankings = st.multiselect(
        "Select Ranking Categories to Display:",
        options=list(ranking_categories.keys())
    )

    # Check if YOY is needed but not available
    if any("Growth" in ranking and "YOY" not in filtered_df_with_ratios.columns for ranking in selected_rankings):
        st.warning(
            "Year-over-Year (YOY) rankings require a valid starting year and a number of years greater than 1. "
            "Please provide these inputs in the sidebar to enable YOY metrics."
    )

    # Display rankings
    for ranking in selected_rankings:
        ranking_info = ranking_categories[ranking]
        sort_col = ranking_info["column"]
        ascending = ranking_info["ascending"]
        display_cols = ranking_info["columns_to_display"]

# Check if the sort column exists in the dataframe
        if sort_col not in filtered_df_with_ratios.columns:
            st.warning(f"Column '{sort_col}' not found for ranking '{ranking}'. Skipping.")
            continue

        # Drop rows with NaN in the sort column
        ranking_df = filtered_df_with_ratios.dropna(subset=[sort_col])

        if ranking_df.empty:
            st.warning(f"No data available for ranking '{ranking}'.")
            continue

        # Sort the dataframe
        ranking_df_sorted = ranking_df.sort_values(by=sort_col, ascending=ascending).head(ranking_limit)

        # Rename columns for better readability
        display_df = ranking_df_sorted[display_cols].copy()
        display_df = display_df.rename(columns={
            "Financial.revenue": "Revenue",
            "REVENUE_GROWTH_PERCENT": "Revenue Growth (%)",
            "REVENUE_GROWTH_NOMINAL": "Revenue Growth (Nominal)",
            "Number.of.employees": "Number of Employees",               
              "EMPLOYEES_GROWTH_PERCENT": "Employees Growth (%)",
                "EMPLOYEES_GROWTH_NOMINAL": "Employees Growth (Nominal)",
                "EBIT": "EBIT",
                "EBIT_GROWTH_PERCENT": "EBIT Growth (%)",
                "EBIT_GROWTH_NOMINAL": "EBIT Growth (Nominal)",
                "PROFIT": "Net Profit",
                "PROFIT_GROWTH_PERCENT": "Profit Growth (%)",
                "PROFIT_GROWTH_NOMINAL": "Profit Growth (Nominal)",
                "Average_salary": "Average Salary",
                "AVG_SALARY_GROWTH_PERCENT": "Salary Growth (%)",
                "AVG_SALARY_GROWTH_NOMINAL": "Salary Growth (Nominal)"
            })

            # Format numerical columns for better readability
        for col in display_df.columns:
            if col != "Name":
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
                if "Growth (%)" in col or "Margin" in col:
                    display_df[col] = display_df[col].map("{:,.2f}".format) + "%"
                elif "Nominal" in col or "Revenue" in col or "EBIT" in col or "Profit" in col or "Salary" in col:
                    display_df[col] = display_df[col].map("{:,.2f}".format)

        st.markdown(f"**{ranking}**")
        st.dataframe(display_df.reset_index(drop=True))

    # ===========================
    # Export Options
    # ===========================
    st.markdown("### Export Full Filtered Data")
    csv_data = filtered_df_with_ratios.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

    def to_excel(df):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(filtered_df_with_ratios)
    st.download_button(
        label="Download Excel",
        data=excel_data,
        file_name='filtered_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.write("*(PDF export not implemented in this example.)*")
    st.success("Report generation complete.")

if __name__ == "__main__":
    main()