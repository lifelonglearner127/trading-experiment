import pandas as pd
import plotly.graph_objects as go
import psycopg2
import streamlit as st

import constants as c
from serializers import DBSerializer

st.set_page_config(layout="wide")

DB_CONFIG = {
    "host": c.DB_HOST,
    "database": c.DB_NAME,
    "user": c.DB_USER,
    "password": c.DB_PASSWORD,
}


def fetch_tables():
    with psycopg2.connect(**DB_CONFIG) as conn:
        query = "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        return pd.read_sql(query, conn)["table_name"].tolist()


def fetch_columns(table_name):
    with psycopg2.connect(**DB_CONFIG) as conn:
        query = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = '{table_name}';
        """
        return pd.read_sql(query, conn)["column_name"].tolist()


def customize_streamlit():
    reduce_header_height_style = """
        <style>
            div.block-container {padding-top:1rem;}
        </style>
    """
    st.markdown(reduce_header_height_style, unsafe_allow_html=True)


def plot_candlestick(df, label_column):
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["timestamp"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )

    for i, row in df.iterrows():
        if row[label_column] != 0:
            text = "O" if row[label_column] == 1 else "X"
            fig.add_trace(
                go.Scatter(
                    x=[row["timestamp"]],
                    y=[(row["open"] + row["close"]) / 2],
                    mode="text",
                    text=text,
                    textfont=dict(color="red", size=14),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"Candlestick Chart for {label_column}",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
    )
    return fig


customize_streamlit()

st.title("CandleStick Chart Viewer")
st.sidebar.header("Table Settings")
table_list = fetch_tables()
selected_table = st.sidebar.selectbox("Select a Table", table_list)

if selected_table:
    column_list = fetch_columns(selected_table)
    label_columns = [column for column in column_list if column.startswith("label")]
    label_column = st.sidebar.selectbox("Select a Label Column", label_columns)

    status_placeholder = st.empty()
    if st.sidebar.button("Fetch and Plot Data"):
        status_placeholder.text(f"Fetching data from table `{selected_table}`...")
        db_serializer = DBSerializer(c.DB_URI)
        df = db_serializer.load_from_db(selected_table)
        status_placeholder.text(f"Displaying data for table: {selected_table}")

        if label_column:
            fig = plot_candlestick(df, label_column)
            status_placeholder.empty()
            st.plotly_chart(fig)
