
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="House Price EDA Dashboard",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 House Price EDA Dashboard")
st.write("Explore and analyze house prices using Python.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

df = load_data()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a section",
    ["Overview", "Price Analysis", "Data Analysis", "Missing Values"]
)

if page == "Overview":
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Houses", len(df))
    col2.metric("Total Features", len(df.columns))
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

elif page == "Price Analysis":
    st.header("House Price Analysis")

    if "SalePrice" in df.columns:
        col1, col2, col3 = st.columns(3)
        col1.metric("Average Price", f"${df['SalePrice'].mean():,.0f}")
        col2.metric("Minimum Price", f"${df['SalePrice'].min():,.0f}")
        col3.metric("Maximum Price", f"${df['SalePrice'].max():,.0f}")

        st.subheader("Sale Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["SalePrice"], kde=True, ax=ax)
        ax.set_xlabel("Sale Price")
        st.pyplot(fig)

        st.subheader("Sale Price Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(x=df["SalePrice"], ax=ax)
        st.pyplot(fig)
    else:
        st.warning("SalePrice column not found.")

elif page == "Data Analysis":
    st.header("Feature Analysis")

    numeric_df = df.select_dtypes(include="number")
    selected_column = st.selectbox(
        "Choose a feature",
        numeric_df.columns
    )

    st.subheader(f"Distribution of {selected_column}")
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
    st.pyplot(fig)

    if "SalePrice" in numeric_df.columns:
        st.subheader("Correlation with Sale Price")
        correlation = numeric_df.corr()["SalePrice"].sort_values(
            ascending=False
        )
        st.bar_chart(correlation.drop("SalePrice", errors="ignore"))

elif page == "Missing Values":
    st.header("Missing Values Analysis")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) > 0:
        st.dataframe(
            missing.rename("Missing Count"),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        missing.head(20).plot(kind="bar", ax=ax)
        ax.set_ylabel("Number of Missing Values")
        ax.set_title("Top 20 Columns with Missing Values")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    else:
        st.success("No missing values found.")

st.sidebar.markdown("---")
st.sidebar.caption("Built with Python, Pandas, Matplotlib and Streamlit")
