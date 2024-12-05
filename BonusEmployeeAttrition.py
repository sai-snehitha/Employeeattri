import streamlit as st
import psycopg2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Database connection setup
def connect_to_database():
    try:
        conn = psycopg2.connect(
            host="employee-attrition-db-snehitha2303-34c3.g.aivencloud.com",
            database="defaultdb",
            user="avnadmin",
            password="AVNS_7dkrpFMqyM9xA0Wr2CR",
            port="16750",
            sslmode="require"
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"Cannot connect to the database: {e}")
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
            return f"Execution query error: {e}"
        finally:
            conn.close()
    else:
        return None

# Sidebar
st.sidebar.title("Attrition Insights Portal")
st.sidebar.markdown("Navigate through the options below:")

options = st.sidebar.selectbox(
    "Select an Option",
    [
        "Home",
        "Query Attrition Details",
        "Query Commute and Absenteeism",
        "Employee Details",
        "Predictive Analysis",
        "Run Custom Query"
    ]
)

# 1. Home Section
if options == "Home":
    st.title("Welcome to Employee Attrition Insights Portal")
    st.markdown("""
    This application helps businesses analyze employee attrition and performance trends.
    Use the sidebar to navigate through features like Predictive Analysis, Custom Queries, and more.
    """)

    # Test database connection
    st.markdown("Testing database connection...")
    conn = connect_to_database()
    if conn:
        st.success("Successfully connected to the database!")
        conn.close()
    else:
        st.error("Failed to connect to the database. Check connection settings.")

# 2. Query Attrition Details
elif options == "Query Attrition Details":
    st.title("Attrition Details")
    st.markdown("Query employee attrition data by job role id.")

    # User input for job role
    jobrole_id = st.number_input("Enter Job Role ID", min_value=1, step=1)
    query = f"""
    SELECT 
        a.jobroleid, 
        j.department, 
        j.job_role, 
        a.reason_for_leaving, 
        a.company_culture_fit, 
        a.team_dynamics, 
        a.company_loyalty
    FROM attrition a
    JOIN jobrole j ON a.jobroleid = j.jobroleid
    WHERE a.jobroleid = {jobrole_id};
    """

    if st.button("Execute Query"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        elif isinstance(result, pd.DataFrame) and result.empty:
            st.warning("No records found for the given Job Role ID.")
        else:
            st.error(result)

# 3. Query Commute and Absenteeism
elif options == "Query Commute and Absenteeism":
    st.title("Commute and Absenteeism Data")
    st.markdown("View commute distance and absenteeism rates for employees.")

    query = """
    SELECT 
        c.employee_id, 
        e.age, 
        e.gender, 
        c.commute_distance, 
        c.absenteeism_rate
    FROM commuteandabsenteeism c
    JOIN employee e ON c.employee_id = e.employee_id;
    """

    if st.button("Fetch Commute Data"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        else:
            st.warning("No commute and absenteeism data found.")

# 4. Employee Details
elif options == "Employee Details":
    st.title("Search Employee Details")
    st.markdown("Search for an employee by their ID and view all available details.")

    # Input for employee search
    emp_id = st.text_input("Enter Employee ID(Ex: E00001, E00002...)")

    query = f"""
    SELECT 
        e.employee_id, 
        e.age, 
        e.gender, 
        e.marital_status, 
        j.department, 
        j.job_role, 
        j.salary
    FROM employee e
    JOIN jobrole j ON e.jobroleid = j.jobroleid
    WHERE e.employee_id = '{emp_id}';
    """

    if st.button("Search Employee"):
        result = execute_query(query)
        if isinstance(result, pd.DataFrame) and not result.empty:
            st.dataframe(result)
        else:
            st.warning("No employee details found for the given Employee ID.")

# 5. Predictive Analysis
elif options == "Predictive Analysis":
    st.title("Predictive Analysis")
    st.markdown("Analyze and visualize employee data trends using plots and models.")

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
        x_axis = st.selectbox("Select Attribute1:", numeric_columns)
        y_axis = st.selectbox("Select Attribute2:", numeric_columns)

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
        feature = st.selectbox("Select Feature Variable (X):", numeric_columns)

        if target and feature:
            X = df[[feature]]
            y = df[target]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if st.button("Result"):
                # Train linear regression model
                model = LinearRegression()
                model.fit(X_train, y_train)

                # Make predictions
                y_pred = model.predict(X_test)

                # Display metrics
                st.markdown("### Model Performance")
                st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    else:
        st.error("Failed to load data. Please check the database or query.")

# 6. Run Custom Query
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
