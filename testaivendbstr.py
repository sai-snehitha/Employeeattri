import streamlit as st
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Database connection setup
def connect_to_database():
    try:
        conn = psycopg2.connect(
            host="employee-attrition-db-snehitha2303-34c3.g.aivencloud.com",  # Aiven PostgreSQL host
            database="defaultdb",  # Aiven database name
            user="avnadmin",  # Aiven username
            password="AVNS_7dkrpFMqyM9xA0Wr2CR",  # Aiven password
            port="16750",  # Aiven port
            sslmode="require"  # Ensure SSL is used
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Error connecting to the database: {e}")
        return None

# Execute query function
def execute_query(query):
    conn = connect_to_database()
    if conn:
        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                if cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return pd.DataFrame(rows, columns=columns)
                else:
                    conn.commit()
                    return "Query executed successfully."
        except Exception as e:
            st.error(f"Error executing query: {e}")
            return None
        finally:
            conn.close()
    else:
        return None

# Sidebar Navigation
st.sidebar.title("Employee Attrition Management")
st.sidebar.markdown("Navigate through the options below:")

options = st.sidebar.selectbox(
    "Select an Option",
    [
        "Home",
        "Predictive Analysis",
        "Run Custom Query"
    ]
)

# Home Section
if options == "Home":
    st.title("Welcome to the Employee Attrition Management System")
    st.markdown("Use the sidebar to navigate through different features.")

    # Test database connection
    st.markdown("Testing database connection...")
    conn = connect_to_database()
    if conn:
        st.success("Successfully connected to the PostgreSQL database!")
        conn.close()
    else:
        st.error("Failed to connect to the database. Check connection settings.")

# Predictive Analysis Section
elif options == "Predictive Analysis":
    st.title("Predictive Analysis")
    st.markdown("Analyze and visualize data trends using different plots and models.")

    # Load data from the database
    query = """
    SELECT 
        employee.age, 
        jobrole.salary, 
        performance.performance_rating, 
        performance.work_life_balance_rating, 
        training.training_hours,
        performance.promotion_count,
        performance.years_at_company
    FROM employee
    JOIN jobrole ON employee.jobroleid = jobrole.jobroleid
    JOIN performance ON employee.employee_id = performance.employee_id
    JOIN training ON employee.jobroleid = training.jobroleid;
    """
    df = execute_query(query)

    if isinstance(df, pd.DataFrame) and not df.empty:
        st.markdown("### Dataset Overview")
        st.write(df.head())

        # Select columns for visualization
        numeric_columns = df.select_dtypes(include=['number']).columns
        x_axis = st.selectbox("Select X-Axis:", numeric_columns)
        y_axis = st.selectbox("Select Y-Axis:", numeric_columns)

        # Select plot type
        plot_type = st.radio("Select Plot Type:", ["Scatter Plot", "Line Plot", "Histogram", "Bar Plot"])

        # Generate plot based on user input
        st.markdown("### Visualization")
        plt.figure(figsize=(10, 5))

        if plot_type == "Scatter Plot":
            sns.scatterplot(data=df, x=x_axis, y=y_axis)
        elif plot_type == "Line Plot":
            sns.lineplot(data=df, x=x_axis, y=y_axis)
        elif plot_type == "Histogram":
            sns.histplot(data=df, x=x_axis, bins=30, kde=True)
        elif plot_type == "Bar Plot":
            sns.barplot(data=df, x=x_axis, y=y_axis)

        st.pyplot(plt)

        # Predictive Model
        st.markdown("### Build Predictive Model")
        target = st.selectbox("Select Target Variable (Y):", numeric_columns)
        features = st.multiselect("Select Feature Variables (X):", numeric_columns)

        if target and features:
            X = df[features]
            y = df[target]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train linear regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Display metrics
            st.markdown("### Model Performance")
            st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
            st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    else:
        st.error("Failed to load data. Please check the database or query.")

# Run Custom Query Section
elif options == "Run Custom Query":
    st.title("Run Custom SQL Query")
    st.markdown("Write your SQL query below and click 'Run Query' to execute it on the database.")

    # Input textbox for custom query
    custom_query = st.text_area("Enter your SQL query here", height=150)

    if st.button("Run Query"):
        if custom_query.strip():
            result = execute_query(custom_query)
            if isinstance(result, pd.DataFrame) and not result.empty:
                st.dataframe(result)
            elif isinstance(result, pd.DataFrame) and result.empty:
                st.warning("The query executed successfully but returned no results.")
            else:
                st.error(result)
        else:
            st.error("Please enter a valid SQL query.")
